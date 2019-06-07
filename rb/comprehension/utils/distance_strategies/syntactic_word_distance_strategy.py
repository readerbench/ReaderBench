from rb.comprehension.utils.distance_strategies.cm_syntactic_graph import CmSyntacticGraph
from rb.comprehension.utils.graph.cm_edge_do import CmEdgeDO
from rb.comprehension.utils.graph.cm_edge_type import CmEdgeType
from rb.comprehension.utils.distance_strategies.abstract_distance_strategy import DistanceStrategy


from rb.core.word import Word


class SyntacticWordDistanceStrategy(DistanceStrategy):

    def __init__(self, syntactic_graph: CmSyntacticGraph):
        self.edge_list = syntactic_graph.edge_list

    
    def get_distance(self, first_word: Word, second_word: Word) -> float:
        for edge in self.edge_list:
            dependent_word = edge.node1.word
            governor_word = edge.node2.word

            if (dependent_word == first_word and governor_word == second_word) or \
                (dependent_word == second_word and governor_word == first_word):
                return 1.0
        
        return 0.0


    def get_edge_type(self) -> CmEdgeType:
        return CmEdgeType.Syntactic