from rb.comprehension.utils.graph.cm_node_do import CmNodeDO
from rb.comprehension.utils.graph.cm_edge_do import CmEdgeDO
from rb.comprehension.utils.graph.cm_node_type import CmNodeType
from rb.comprehension.utils.graph.cm_edge_type import CmEdgeType
from rb.core.pos import POS
from rb.core.word import Word
from rb.core.lang import Lang
from rb.core.text_element import TextElement
from rb.core.sentence import Sentence
from rb.similarity.vector_model import VectorModel
from rb.similarity.aoa import AgeOfAcquisition
from copy import deepcopy
from typing import List, Dict
import rb.similarity.wordnet as wordnet
import numpy as np

Nodes = List[CmNodeDO]
Edges = List[CmEdgeDO]
Models = List[VectorModel]

import time


class CmGraphDO:

    def __init__(self, node_list: Nodes, edge_list: Edges) -> None:
        self.node_list = node_list
        self.edge_list = edge_list
        self.adjacent_edges_dict = {}
        self.init_adjacent_edges_dict()

    def init_adjacent_edges_dict(self) -> None:
        for edge in self.edge_list:
            self.adjacent_edges_dict[edge.node1] = []
            self.adjacent_edges_dict[edge.node2] = []

    def contains_node(self, node: CmNodeDO) -> bool:
        return node in self.node_list

    def contains_edge(self, edge: CmEdgeDO) -> bool:
        return edge in self.edge_list

    def get_node(self, node: CmNodeDO) -> CmNodeDO:
        for in_node in self.node_list:
            if in_node == node:
                return in_node
        return None

    def get_edge(self, edge: CmEdgeDO) -> CmEdgeDO:
        for in_edge in self.edge_list:
            if in_edge == edge:
                return in_edge
        return None

    def remove_node_links(self, node: CmNodeDO) -> None:
        if not self.contains_node(node):
            return

        graph_node = self.get_node(node)

        node_edges = []
        for edge in self.edge_list:
            if edge.get_opposite_node(graph_node):
                node_edges.append(edge)

        for edge in node_edges:
            self.edge_list.remove(edge)

    def add_node_or_update(self, node: CmNodeDO) -> None:
        if not self.contains_node(node):
            self.node_list.append(node)
            self.adjacent_edges_dict[node] = []
            return

        graph_node = self.get_node(node)
        graph_node.activation_score += node.activation_score
        graph_node.deactivate()
        if node.is_active():
            graph_node.activate()

        # if the other node is inferred we keep the graph node as it is
        if node.node_type == CmNodeType.TextBased:
            graph_node.node_type = CmNodeType.TextBased

    def add_edge_or_update(self, edge: CmEdgeDO) -> None:
        if not self.contains_edge(edge):
            self.edge_list.append(edge)
            self.adjacent_edges_dict[edge.node1].append(edge)
            self.adjacent_edges_dict[edge.node2].append(edge)
            return

        graph_edge = self.get_edge(edge)
        graph_edge.deactivate()
        if edge.is_active():
            graph_edge.activate()

    def get_edges_for_node(self, node: CmNodeDO) -> Edges:
        return self.adjacent_edges_dict[node]

    def get_activate_edges_for_node(self, node: CmNodeDO) -> Edges:
        return [e for e in self.adjacent_edges_dict[node] if e.is_active()]

    def restrict_active_nodes(self, max_active_concepts: int) -> None:
        self.node_list.sort(key=lambda x: x.activation_score, reverse=True)

        partial_node_list = self.node_list[max_active_concepts:]

        for node in partial_node_list:
            if node.is_active():
                node.deactivate()
                node_edges = self.get_edges_for_node(node)
                for edge in node_edges:
                    edge.deactivate()

    def set_node_list(self, node_list: Nodes) -> None:
        self.node_list = node_list

    def set_edge_list(self, edge_list: Nodes) -> None:
        self.edge_list = edge_list

    def combine_links_from_graph(self, other_graph: 'CmGraphDO') -> None:
        for edge in other_graph.edge_list:
            if not self.contains_edge(edge):
                self.add_node_or_update(edge.node1)
                self.add_node_or_update(edge.node2)
                self.add_edge_or_update(edge)

    def avg_similarity_using_models(self, word1: Word, word2: Word, semantic_models: Models) -> float:
        avg_models_similarity = 0.0
        for sm in semantic_models:
            avg_models_similarity += sm.similarity(word1, word2)
        if semantic_models:
            avg_models_similarity /= len(semantic_models)
        return avg_models_similarity

    def compute_similarity_in_all_models(self, e1: TextElement, e2: TextElement, semantic_models: Models) -> float:
        sum_of_similarities = 0
        for model in semantic_models:
            sum_of_similarities += model.similarity(e1, e2)
        if len(semantic_models) > 0:
            mean = sum_of_similarities / len(semantic_models)
            return mean
        return 0

    def refine_similar_concepts(self, sentence: Sentence, similar_concepts: List[str],
                                lang: Lang, semantic_models: Models, aoa: AgeOfAcquisition) -> List[str]:
        return [concept for concept in similar_concepts
                if self.compute_similarity_in_all_models(sentence, Word.from_str(lang, concept), semantic_models) > 0.33
                and aoa.get_kuperman_value(concept) and aoa.get_kuperman_value(concept) < 9]

    def combine_with_syntactic_links(self, syntactic_graph: 'CmGraphDO', sentence: Sentence,
                                     semantic_models: Models, max_dictionary_expansion: int) -> None:

        inferred_nodes = set()

        aoa = None
        for node in syntactic_graph.node_list:
            node.activate()
            node.increment_activation_score()

            if not aoa:
                aoa = AgeOfAcquisition(node.get_word().lang)

            self.add_node_or_update(node)

            if node.get_word().pos != POS.NOUN and node.get_word().pos != POS.VERB:
                continue

            synonyms = wordnet.get_synonyms(node.get_word())
            hypernyms = wordnet.get_hypernyms(node.get_word())
            # print("Wordnet {}".format(int(t_wn_e - t_wn_s)))
            similar_concepts = []
            similar_concepts.extend(synonyms)
            similar_concepts.extend(hypernyms)

            for vect_model in semantic_models:
                closest_semantic_words = vect_model.most_similar(node.get_word(), topN=5, threshold=0.5)
                similar_concepts.extend([x[0] for x in closest_semantic_words])
            # print("Vector models {}".format(int(t_vm_e - t_vm_s)))
            similar_concepts = list(set(similar_concepts))
            # remove the word if that is the case
            similar_concepts = [x for x in similar_concepts if x != node.get_word().lemma]
            # print("before", similar_concepts)
            similar_concepts = self.refine_similar_concepts(sentence, similar_concepts, node.get_word().lang,
                                                            semantic_models, aoa)
            # print("Rafinare {}".format(int(t_refine_e - t_refine_s)))
            # print("after", similar_concepts)

            for concept in similar_concepts:
                word = Word.from_str(node.get_word().lang, concept, node.get_word().pos)
                inferred_node = CmNodeDO(word, CmNodeType.Inferred)
                inferred_node.activate()
                inferred_nodes.add(inferred_node)

        for edge in syntactic_graph.edge_list:
            self.add_edge_or_update(edge)

        # deactivate all the semantic links
        for edge in self.edge_list:
            if edge.edge_type == CmEdgeType.Semantic:
                edge.deactivate()

        # get all potential inferred edge list
        potential_inferred_node_list = []

        for inferred_node in inferred_nodes:
            avg_similarity = 0.0
            for syntactic_node in syntactic_graph.node_list:
                avg_similarity += self.avg_similarity_using_models(inferred_node.word,
                                                                   syntactic_node.word, semantic_models)

            if syntactic_graph.node_list:
                avg_similarity /= len(syntactic_graph.node_list)

            inferred_node.activation_score = avg_similarity
            potential_inferred_node_list.append(inferred_node)

        # sort potential inferred nodes
        potential_inferred_node_list.sort(key=lambda x: x.activation_score, reverse=True)

        inferred_node_list = potential_inferred_node_list[
                             0: min(max_dictionary_expansion, len(potential_inferred_node_list))]

        for node in inferred_node_list:
            self.add_node_or_update(node)

        active_nodes = [node for node in self.node_list if node.is_active()]
        N = len(active_nodes)

        distances = [0] * (N * (N - 1))
        potential_edge_list = []
        for i in range(N):
            node1 = active_nodes[i]
            for j in range(i + 1, N):
                node2 = active_nodes[j]
                similarity = self.avg_similarity_using_models(node1.word, node2.word, semantic_models)
                distances[i * N + j] = similarity
                edge = CmEdgeDO(node1, node2, CmEdgeType.Semantic, similarity)
                edge.activate()
                potential_edge_list.append(edge)

        avg = np.average(distances)
        std = np.std(distances)
        min_distance = min(0.3, avg + std)

        for edge in potential_edge_list:
            if edge.score >= min_distance:
                self.add_edge_or_update(edge)

    def get_combined_graph(self, other_graph: 'CmGraphDO') -> 'CmGraphDO':
        new_node_list = deepcopy(self.node_list)
        for node in other_graph.node_list:
            if node not in new_node_list:
                new_node_list.append(node)

        new_edge_list = deepcopy(self.edge_list)
        for edge in other_graph.edge_list:
            if edge not in new_edge_list:
                new_edge_list.append(edge)

        graph = CmGraphDO(new_node_list, new_edge_list)
        return graph

    def get_activation_map(self) -> Dict[CmNodeDO, float]:
        activation_map = {}
        for node in self.node_list:
            activation_map[node] = node.activation_score
        return activation_map

    def __repr__(self):
        return str(self.node_list) + '\n' + str(self.edge_list)

    def __str__(self):
        return str(self.node_list) + '\n' + str(self.edge_list)
