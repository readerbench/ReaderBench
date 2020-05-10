from rb.comprehension.utils.graph.cm_node_do import CmNodeDO, CmNodeType
from rb.comprehension.utils.memory.word_activation import WordActivation
from rb.comprehension.utils.graph.cm_graph_do import CmGraphDO
from rb.comprehension.utils.graph.graph_metrics import GraphMetrics

import numpy as np

from typing import Dict

ActivationMap = Dict[CmNodeDO, float]


class HistoryKeeper:

    def __init__(self):
        self.activation_history = []
        self.graph_metrics_history = []
        self.unique_word_list = []

    def save_state(self, activation_map: ActivationMap, graph: CmGraphDO):
        self.save_scores(activation_map)
        self.graph_metrics_history.append(GraphMetrics(graph))

    def save_scores(self, activation_map: ActivationMap) -> None:
        activation_auxiliary = {}

        for key in activation_map.keys():
            activation_auxiliary[key.word] = WordActivation(activation_map[key], key.active)

        self.activation_history.append(activation_auxiliary)

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

    def compute_statistics(self):
        mean_activation_for_active_nodes = self.compute_mean_activation_for_active_nodes()
        mean_activation_for_all_nodes = self.compute_mean_activation_for_all_nodes()
        mean_degree_centrality = self.compute_mean_degree_centrality()
        mean_closeness_centrality = self.compute_mean_closeness_centrality()
        mean_betweenness_centrality = self.compute_mean_betweenness_centrality()
        mean_harmonic_centrality = self.compute_mean_harmonic_centrality()
        mean_active_nodes_percentage = self.compute_mean_active_nodes_percentage()
        mean_density = self.compute_mean_density()
        mean_modularity = self.compute_mean_modularity()

        return [mean_activation_for_active_nodes, mean_activation_for_all_nodes, mean_degree_centrality,
                mean_closeness_centrality, mean_betweenness_centrality, mean_harmonic_centrality,
                mean_active_nodes_percentage, mean_density, mean_modularity]

    def compute_mean_activation_for_active_nodes(self):
        total_activation_sum = 0.0
        total_count = len(self.activation_history)
        for activation_dictionary in self.activation_history:
            total_activation_sum += np.mean([word_activation.activation_value
                                             for word_activation in activation_dictionary.values()
                                             if word_activation.active])

        return total_activation_sum / total_count

    def compute_mean_activation_for_all_nodes(self):
        total_activation_sum = 0.0
        total_count = len(self.activation_history)
        for activation_dictionary in self.activation_history:
            total_activation_sum += np.mean([word_activation.activation_value
                                             for word_activation in activation_dictionary.values()])

        return total_activation_sum / total_count

    def compute_mean_centrality(self, centrality_function):
        total_centrality = 0.0
        total_count = len(self.graph_metrics_history)
        for graph_metrics in self.graph_metrics_history:
            method = getattr(graph_metrics, centrality_function, None)
            centrality_dictionary = method()
            total_centrality += np.mean(list(centrality_dictionary.values()))

        return total_centrality / total_count

    def compute_mean_degree_centrality(self):
        return self.compute_mean_centrality("degree_centrality")

    def compute_mean_closeness_centrality(self):
        return self.compute_mean_centrality("closeness_centrality")

    def compute_mean_betweenness_centrality(self):
        return self.compute_mean_centrality("betweenness_centrality")

    def compute_mean_harmonic_centrality(self):
        return self.compute_mean_centrality("harmonic_centrality")

    def compute_mean_number_of_active_nodes(self):
        return np.mean([len(gm.G.nodes) for gm in self.graph_metrics_history])

    def compute_mean_number_of_active_edges(self):
        return np.mean([len(gm.G.edges) for gm in self.graph_metrics_history])

    def compute_mean_active_nodes_percentage(self):
        return np.mean([len(graph_metrics.G.nodes) / (len(self.activation_history[index]))
                        for index, graph_metrics in enumerate(self.graph_metrics_history)])

    def compute_mean_metrics(self, metric_function):
        return np.mean([getattr(graph_metrics, metric_function, None)()
                        for graph_metrics in self.graph_metrics_history])

    def compute_mean_density(self):
        return self.compute_mean_metrics("density")

    def compute_mean_modularity(self):
        return self.compute_mean_metrics("modularity")

