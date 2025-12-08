
from rb.core.word import Word
from rb.comprehension.utils.distance_strategies.abstract_distance_strategy import DistanceStrategy
from rb.comprehension.utils.graph.cm_node_type import CmNodeType
from rb.comprehension.utils.graph.cm_graph_do import CmGraphDO
from rb.comprehension.utils.graph.cm_node_do import CmNodeDO
from rb.comprehension.utils.graph.cm_edge_do import CmEdgeDO

import numpy as np

from typing import List
Words = List[Word]


class CmWordDistanceIndexer():

    def __init__(self, word_list: Words, strategy: DistanceStrategy):
        self.word_list = word_list
        self.strategy = strategy
        self.index_distances()


    def index_distances(self) -> None:
        n = len(self.word_list)
        self.distances = np.zeros((n, n))

        for i, word1 in enumerate(self.word_list):
            for j, word2 in enumerate(self.word_list):
                if i == j:
                    self.distances[i][j] = self.distances[j][i] = 1
                else:
                    self.distances[i][j] = self.distances[j][i] = self.strategy.get_distance(word1, word2)

        
    def cut_by_avg_plus_stdev(self, minimum_distance: float) -> None:
        threshold = self.get_avg_plus_stdev_threshold(minimum_distance)

        new_word_list = [word for index, word in enumerate(self.word_list)
                            if self.get_max_distance_value_for_word_at_line(index) >= threshold]
        
        self.word_list = new_word_list
        self.index_distances()


    def get_avg_plus_stdev_threshold(self, minimum_distance: float) -> float:
        ok_distances = [self.distances[i][j]
                        for i in range(len(self.word_list) - 1) 
                        for j in range(i + 1, len(self.word_list)) 
                        if self.distances[i][j] > minimum_distance]
        
        if ok_distances:
            avg = np.mean(ok_distances)
            std = np.std(ok_distances)
            return avg - std
        return 0.0


    def get_max_distance_value_for_word_at_line(self, line_number: int) -> float:
        return np.max(self.distances[line_number])

    
    def get_cm_graph(self, node_type: CmNodeType) -> CmGraphDO:
        graph = CmGraphDO([], [])

        node_set = set()
        n = len(self.word_list)
        for i in range(n - 1):
            for j in range(i + 1, n):
                if i != j and self.distances[i][j] > 0:
                    w1 = self.word_list[i]
                    w2 = self.word_list[j]
                    
                    n1 = CmNodeDO(w1, node_type)
                    n1.activate()

                    n2 = CmNodeDO(w2, node_type)
                    n2.activate()

                    edge = CmEdgeDO(n1, n2, self.strategy.get_edge_type(), self.distances[i][j])

                    graph.edge_list.append(edge)
                    node_set.add(n1)
                    node_set.add(n2)

        graph.set_node_list([node for node in node_set if node.word.is_content_word()])

        return graph