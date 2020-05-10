from rb.core.sentence import Sentence
from rb.core.word import Word
from rb.comprehension.utils.graph.cm_graph_do import CmGraphDO
from rb.comprehension.utils.graph.cm_node_do import CmNodeDO
from rb.comprehension.utils.graph.cm_node_type import CmNodeType
from rb.comprehension.utils.graph.cm_edge_do import CmEdgeDO, CmEdgeType

from typing import List, Dict, Set

class CmDependencyGraf:

    def __init__(self):
        self.node_list: Set[Word] = set([])
        self.adjacent_list: Dict[Word, Set[Word]] = {}

    def get_actual_word(self, word):
        if word.in_coref:
            return word.coref_clusters[0].main.get_root()
        return word

    def build_graph(self, sentence: Sentence) -> None:
        for index, dependency in enumerate(sentence.get_dependencies()):
            w0 = self.get_actual_word(dependency[0])
            w1 = self.get_actual_word(dependency[1])
            self.node_list.add(w0)
            self.node_list.add(w1)
            if w0 in self.adjacent_list:
                self.adjacent_list[w0].add(w1)
            else:
                self.adjacent_list[w0] = set([w1])

            if w1 in self.adjacent_list:
                self.adjacent_list[w1].add(w0)
            else:
                self.adjacent_list[w1] = set([w0])

    def dfs(self, node: Word) -> List[Word]:
        results = []
        stack = []
        visited = set([])
        
        stack.append(node)
        visited.add(node)
        while stack:
            word = stack.pop()
            for neighbour in self.adjacent_list[word]:
                if neighbour not in visited:
                    if neighbour.is_content_word():
                        results.append(neighbour)
                    else:
                        stack.append(neighbour)
                    visited.add(neighbour)

        return results

    def get_syntactic_graph(self, sentence: Sentence) -> CmGraphDO:
        self.build_graph(sentence)
        graph = CmGraphDO([], [])
        node_set = set([])
        for node in self.node_list:
            if node.is_content_word():
                accessible_content_words = self.dfs(node)
                n1 = CmNodeDO(node, CmNodeType.TextBased)
                n1.activate()
                node_set.add(n1)
                for word in accessible_content_words:
                    n2 = CmNodeDO(word, CmNodeType.TextBased)
                    n2.activate()
                    node_set.add(n2)

                    edge = CmEdgeDO(n1, n2, CmEdgeType.Syntactic, 1)

                    if edge not in graph.edge_list:
                        graph.edge_list.append(edge)
        
        graph.node_list = list(node_set)

        return graph
