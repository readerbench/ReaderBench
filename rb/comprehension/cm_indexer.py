from rb.similarity.vector_model import VectorModel
from rb.core.document import Document
from rb.core.lang import Lang
from rb.comprehension.utils.distance_strategies.cm_coref_indexer import CmCorefIndexer
from rb.comprehension.utils.distance_strategies.syntactic_word_distance_strategy import SyntacticWordDistanceStrategy
from rb.comprehension.cm_word_distance_indexer import CmWordDistanceIndexer
from typing import List
Models = List[VectorModel]


class CmIndexer():

    def __init__(self, text: str, lang: Lang, semantic_models: Models):
        self.text = text
        self.semantic_models = semantic_models
        self.lang = lang
        self.document = Document(lang, text)
        self.syntactic_indexer_list = []
        self.index_syntactic_distances()

    def index_syntactic_distances(self) -> None:
        coref_container = CmCorefIndexer(self.document, self.lang)
        sentences = self.document.get_sentences()
        sentence_number = 0
        for sentence in sentences:
            syntactic_graph = coref_container.get_cm_syntactic_graph(sentence, sentence_number)
            syntactic_strategy = SyntacticWordDistanceStrategy(syntactic_graph)
            wd_indexer = CmWordDistanceIndexer(list(syntactic_graph.word_set), syntactic_strategy)
            self.syntactic_indexer_list.append(wd_indexer)
            sentence_number += 1
