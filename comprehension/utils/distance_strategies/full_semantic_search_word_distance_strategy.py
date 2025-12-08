from rb.comprehension.utils.graph.cm_edge_type import CmEdgeType
from rb.comprehension.utils.distance_strategies.abstract_distance_strategy import DistanceStrategy

from rb.core.word import Word

from rb.similarity.vector_model import VectorModel

from typing import List
Models = List[VectorModel] 


class FullSemanticSpaceWordDistanceStrategy(DistanceStrategy):

    def __init__(self, semantic_models: Models, threshold: float):
        self.semantic_models = semantic_models
        self.threshold = threshold


    def get_distance(self, first_word: Word, second_word: Word) -> float:
        if not self.semantic_models:
            return 0.0

        avg = 0.0
        for sm in self.semantic_models:
            avg += sm.similarity(first_word, second_word)
        avg /= len(sm)

        if avg > self.threshold:
            return avg
        return 0.0

    
    def get_edge_type(self) -> CmEdgeType:
        return CmEdgeType.Semantic