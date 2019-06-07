from rb.core.word import Word
from rb.comprehension.utils.graph.cm_edge_type import CmEdgeType

class DistanceStrategy():

    def get_distance(self, first_word: Word, second_word: Word) -> float:
        pass

    def get_edge_type(self) -> CmEdgeType:
        pass