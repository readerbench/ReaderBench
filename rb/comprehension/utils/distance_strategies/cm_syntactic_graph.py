from rb.comprehension.utils.graph.cm_edge_do import CmEdgeDO
from rb.comprehension.utils.graph.cm_edge_type import CmEdgeType
from rb.comprehension.utils.graph.cm_node_do import CmNodeDO
from rb.comprehension.utils.graph.cm_node_type import CmNodeType

from rb.core.word import Word


class CmSyntacticGraph():

    def __init__(self):
        self.edge_list = []
        self.word_set = set()

    
    def index_edge(self, word1: Word, word2: Word) -> None:
        node1 = CmNodeDO(word1, CmNodeType.TextBased)
        node2 = CmNodeDO(word2, CmNodeType.TextBased)
        edge = CmEdgeDO(node1, node2, CmEdgeType.Syntactic, 1.0)

        if edge not in self.edge_list:
            self.edge_list.append(edge)
            self.word_set.add(word1)
            self.word_set.add(word2)
