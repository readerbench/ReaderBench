from rb.comprehension.utils.graph.cm_graph_do import CmGraphDO
from rb.comprehension.utils.graph.cm_edge_do import CmEdgeDO
from rb.comprehension.utils.graph.cm_node_do import CmNodeDO

import networkx as nx


class GraphMetrics:

    def __init__(self, graph: CmGraphDO):
        self.cm_graph = graph
        self.epsilon = 0.00001
        self.build_networkx_graph()

    def build_networkx_graph(self):
        self.G = nx.Graph()

        for node in self.cm_graph.node_list:
            if node.is_active():
                self.G.add_node(node.word.lemma)

        for edge in self.cm_graph.edge_list:
            if edge.is_active():
                self.G.add_edge(edge.node1, edge.node2, weight=1 - edge.score if edge.score < 1 else self.epsilon)

    def degree_centrality(self):
        return nx.algorithms.centrality.degree_centrality(self.G)

    def closeness_centrality(self):
        return nx.algorithms.centrality.closeness_centrality(self.G)

    def betweenness_centrality(self):
        return nx.algorithms.centrality.betweenness_centrality(self.G)

    def harmonic_centrality(self):
        return nx.algorithms.centrality.harmonic_centrality(self.G)

    def density(self):
        return nx.classes.function.density(self.G)

    def modularity(self):
        return len(list(nx.algorithms.community.modularity_max.greedy_modularity_communities(self.G)))

