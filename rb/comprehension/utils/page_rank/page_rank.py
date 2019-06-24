from typing import Dict

from rb.comprehension.utils.graph.cm_graph_do import CmGraphDO
from rb.comprehension.utils.graph.cm_node_do import CmNodeDO

ActivationMap = Dict[CmNodeDO, float]


class PageRank():

    def __init__(self):
        self.max_iter = 100000
        self.eps = 0.0001
        self.prob = 0.85

    
    def run_page_rank(self, graph: CmGraphDO) -> None:
        page_rank_values = graph.get_activation_map()
        new_page_rank_values = self.run_page_rank_with_activation(page_rank_values, graph)

        for key, value in new_page_rank_values.items():
            node = graph.get_node(key)
            node.set_activation_score(value)


    def run_page_rank_with_activation(self, activation_map: ActivationMap, graph: CmGraphDO) -> ActivationMap:
        iteration = 0
        current_page_rank_values = dict(activation_map)
        while iteration < self.max_iter:
            r = self.calculateR(current_page_rank_values, graph)
            temp_page_rank_values = {}
            done = True

            for node in graph.node_list:
                temp_pr_value = self.compute_temp_page_rank_values(current_page_rank_values, graph, node, r)
                prev_pr_value = self.get_page_rank_value(current_page_rank_values, node, graph)

                temp_page_rank_values[node] = temp_pr_value

                if (temp_pr_value - prev_pr_value) / prev_pr_value >= self.eps:
                    done = False
            
            current_page_rank_values = temp_page_rank_values
            if done:
                break
            
            iteration += 1
        
        return current_page_rank_values


    def calculateR(self, page_rank_values: ActivationMap, graph: CmGraphDO) -> float:
        r = 0
        n = len(graph.node_list)
        for node in graph.node_list:
            node_edge_list = graph.get_activate_edges_for_node(node)
            node_degree = len(node_edge_list)
            node_page_rank_value = self.get_page_rank_value(page_rank_values, node, graph)
            if node_degree > 0:
                r += (1.0 - self.prob) * (node_page_rank_value / n)
            else:
                r += (node_page_rank_value / n)
        return r


    def compute_temp_page_rank_values(self, page_rank_values: ActivationMap,
                                        graph: CmGraphDO, node: CmNodeDO, r: float) -> float:
        res = r
        node_edge_list = graph.get_activate_edges_for_node(node)
        for edge in node_edge_list:
            neighbour = edge.get_opposite_node(node)
            neighbour_edge_list = graph.get_activate_edges_for_node(neighbour)
            normalize =  len(neighbour_edge_list)
            res += self.prob * (self.get_page_rank_value(page_rank_values, neighbour, graph) / normalize)

        return res


    def get_page_rank_value(self, page_rank_values: ActivationMap, node: CmNodeDO, graph: CmGraphDO) -> float:
        if node in page_rank_values.keys():
            return page_rank_values[node]
        
        return 1 / len(graph.node_list)
