from rb.comprehension.utils.graph.cm_edge_type import CmEdgeType
from rb.comprehension.utils.graph.cm_node_do import CmNodeDO


class CmEdgeDO:

    def __init__(self, node1: CmNodeDO, node2: CmNodeDO, edge_type: CmEdgeType,
                 score: float) -> None:
        self.node1 = node1
        self.node2 = node2
        self.edge_type = edge_type
        self.score = score
        self.active = True

    def get_node_1(self) -> CmNodeDO:
        return self.node1

    def get_node_2(self) -> CmNodeDO:
        return self.node2

    def get_edge_type(self) -> CmEdgeType:
        return self.edge_type

    def get_score(self) -> float:
        return self.score

    def is_active(self) -> bool:
        return self.active

    def activate(self) -> None:
        self.active = True

    def deactivate(self) -> None:
        self.active = False

    def get_opposite_node(self, node: CmNodeDO) -> CmNodeDO:
        if self.node1 == node:
            return self.node2
        elif self.node2 == node:
            return self.node1
        return None

    def __repr__(self):
        return self.node1.get_word().lemma + " - " + self.node2.get_word().lemma \
               + " (" + str(self.edge_type) + ")" + ": " + str(self.score) + \
               " " + str(self.active)

    def __str__(self):
        return self.node1.get_word().lemma + " - " + self.node2.get_word().lemma \
               + " (" + str(self.edge_type) + ")" + ": " + str(self.score) + \
               " " + str(self.active)

    def __eq__(self, other):
        if isinstance(other, CmEdgeDO):
            return ((self.node1 == other.node1 and self.node2 == other.node2) \
                    or (self.node2 == other.node1 and self.node1 == other.node2)) \
                   and self.edge_type == other.edge_type and self.score == other.score
        return NotImplemented

    def __hash__(self):
        return hash(tuple(self.node1, self.node2, self.score))
