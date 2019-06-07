from rb.comprehension.utils.graph.cm_node_do import CmNodeDO, CmNodeType
from rb.comprehension.utils.memory.word_activation import WordActivation
from rb.comprehension.utils.graph.cm_graph_do import CmGraphDO

from typing import Dict

ActivationMap = Dict[CmNodeDO, float]


class HistoryKeeper():

    def __init__(self):
        self.activation_history = []
        self.unique_word_list   = []

    
    def save_scores(self, activation_map: ActivationMap) -> None:
        activation_auxiliar = {}

        for key in activation_map.keys():
            activation_auxiliar[key.word] = WordActivation(activation_map[key], key.active)
        
        self.activation_history.append(activation_auxiliar)

    
    def save_nodes(self, graph: CmGraphDO) -> None:
        for node in graph.node_list:
            self.add_node_if_not_existing(node)

    
    def add_node_if_not_existing(self, node: CmNodeDO) -> None:
        exists = False
        for index, current_node in enumerate(self.unique_word_list):
            if current_node == node:
                if current_node.node_type != CmNodeType.TextBased and node.node_type == CmNodeType.TextBased:
                    self.unique_word_list[index] = node
                exists = True
                break

        if not exists:
            self.unique_word_list.append(node)
