from rb.similarity.vector_model import VectorModel
from rb.core.document import Document
from rb.core.sentence import Sentence
from rb.core.lang import Lang
from rb.comprehension.cm_indexer import CmIndexer
from rb.comprehension.cm_word_distance_indexer import CmWordDistanceIndexer
from rb.comprehension.utils.graph.cm_graph_do import CmGraphDO
from rb.comprehension.utils.page_rank.page_rank import PageRank
from rb.comprehension.utils.memory.history_keeper import HistoryKeeper
from typing import List
Models = List[VectorModel]


class ComprehensionModel:

    def __init__(self, text: str, lang: Lang, sem_models: Models,
            min_activation_score: float, max_active_concepts: int,
            max_dictionary_expansion: int) -> None:
        self.min_activation_score = min_activation_score
        self.max_active_concepts = max_active_concepts
        self.max_dictionary_expansion = max_dictionary_expansion
        self.text = text
        self.document = Document(lang, text)
        self.cm_indexer = CmIndexer(text, lang, sem_models)
        self.current_graph = CmGraphDO([], [])
        self.semantic_models = sem_models
        self.history_keeper = HistoryKeeper()


    def get_total_number_of_phrases(self) -> int:
        return len(self.cm_indexer.syntactic_indexer_list)


    def get_sentence_at_index(self, index: int) -> Sentence:
        return self.document.get_sentences()[index]

    
    def get_syntactic_indexer_at_index(self, index: int) -> CmWordDistanceIndexer:
        return self.cm_indexer.syntactic_indexer_list[index]


    def apply_page_rank(self, sentence_index: int) -> None:
        page_rank = PageRank()
        page_rank.run_page_rank(self.current_graph)
        self.normalize_activation_score_map()
        self.activate_words_over_threshold()
        self.current_graph.restrict_active_nodes(self.max_active_concepts)
        self.history_keeper.save_scores(self.current_graph.get_activation_map())
        

    def activate_words_over_threshold(self) -> None:
        for node in self.current_graph.node_list:
            if node.get_activation_score() < self.min_activation_score:
                node.deactivate()
                for edge in self.current_graph.get_edges_for_node(node):
                    edge.deactivate()
            else:
                node.activate()
                # nu ar trebui si activate niste muchii aici?

    # normalization with max value
    def normalize_activation_score_map(self) -> None:
        max_value = max([node.get_activation_score() for node in self.current_graph.node_list if node.is_active()])

        if max_value == 0:
            return

        for node in self.current_graph.node_list:
            if node.is_active():
                node.activation_score /= max_value


    def save_scores(self, syntactic_graph: CmGraphDO):
        self.history_keeper.save_nodes(syntactic_graph)
        self.history_keeper.save_nodes(self.current_graph)

        


