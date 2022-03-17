import networkx as nx
import pandas as pd
from tabulate import tabulate
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from pprint import pprint
import numpy as np


class GraphMetrics:

    def __init__(self, graph: "Graph"):
        self.graph = graph
        self.semantic_distances_between_articles = graph.get_distances_between_articles()
        self.semantic_distances_between_authors = graph.get_distances_between_authors()
        # self.semantic_similarities_between_articles = graph.get_similarities_between_articles()
        # self.semantic_similarities_between_authors = graph.get_similarities_between_authors()

    def compute_articles_closeness(self):
        distances_graph = build_distance_graph(self.semantic_distances_between_articles)
        return dict(nx.closeness_centrality(distances_graph, distance='weight'))

    def compute_articles_betweenness(self):
        distances_graph = build_distance_graph(self.semantic_distances_between_articles)
        return dict(nx.betweenness_centrality(distances_graph, weight='weight', normalized=True))

    def compute_authors_closeness(self, graph_type='bidirectional'):
        distances_graph = build_distance_graph(self.semantic_distances_between_authors, graph_type)
        return dict(nx.closeness_centrality(distances_graph, distance='weight'))

    def compute_authors_betweenness(self, graph_type='bidirectional'):
        distances_graph = build_distance_graph(self.semantic_distances_between_authors, graph_type)
        return dict(nx.betweenness_centrality(distances_graph, weight='weight', normalized=True))

    def get_top_n_articles_by_closeness(self, n):
        closeness = self.compute_articles_closeness()
        articles_closeness = [(k, v) for k, v in sorted(closeness.items(), key=lambda item: item[1], reverse=True)][:n]
        return articles_closeness

    def get_top_n_articles_by_betweenness(self, n):
        betweenness = self.compute_articles_betweenness()
        articles_betweenness = [(k, v)
                                for k, v in sorted(betweenness.items(), key=lambda item: item[1], reverse=True)][:n]
        return articles_betweenness

    def get_top_n_authors_by_closeness(self, n, graph_type='bidirectional'):
        closeness = self.compute_authors_closeness(graph_type)
        authors_closeness = [(k, v) for k, v in sorted(closeness.items(), key=lambda item: item[1], reverse=True)][:n]
        return authors_closeness

    def get_top_n_authors_by_betweenness(self, n, graph_type='bidirectional'):
        betweenness = self.compute_authors_betweenness(graph_type)
        authors_betweenness = [(k, v)
                               for k, v in sorted(betweenness.items(), key=lambda item: item[1], reverse=True)][:n]
        return authors_betweenness

    def perform_articles_agglomerative_clustering(self):
        distance_threshold = self.graph.articles_mean
        return perform_agglomerative_clustering(self.graph.articles_set, self.semantic_distances_between_articles,
                                                distance_threshold)

    def perform_authors_agglomerative_clustering(self):
        distance_threshold = self.graph.authors_mean - self.graph.authors_std
        print(self.graph.authors_mean, self.graph.authors_std)
        return perform_agglomerative_clustering(self.graph.authors_set, self.semantic_distances_between_authors,
                                                distance_threshold)

    def get_top_clusters_of_articles_by_number_of_elements(self, number_of_clusters):
        return self.get_top_clusters_of_articles_by_criteria(number_of_clusters, sort_clusters_by_size, self.graph)

    def get_top_clusters_of_articles_by_degree_of_elements(self, number_of_clusters):
        return self.get_top_clusters_of_articles_by_criteria(number_of_clusters, sort_clusters_by_degree, self.graph)

    def get_top_clusters_of_articles_by_criteria(self, number_of_clusters, criteria_function, graph):
        clusters = self.perform_articles_agglomerative_clustering()
        sorted_clusters = criteria_function(clusters, graph.get_articles_by_type_degree)
        counter = 0
        results = []
        for label, articles in sorted_clusters.items():
            titles = ",\n".join([article.title for article in articles])
            abstracts = "\n".join([article.abstract for article in articles])
            results.append((titles, abstracts))
            counter += 1
            if counter >= number_of_clusters:
                break
        return results


def build_distance_graph(distances_pairs, graph_type='bidirectional'):
    if graph_type == 'bidirectional':
        distance_graph = nx.Graph()
    else:
        distance_graph = nx.DiGraph()
    for pair in distances_pairs:
        entity1 = pair[0]
        entity2 = pair[1]
        distance = pair[2]
        distance_graph.add_node(entity1)
        distance_graph.add_node(entity2)
        distance_graph.add_edge(entity1, entity2, weight=distance)
    return distance_graph


def build_position_dictionary(entity_set):
    index = 0
    position_dictionary = {}
    for entity in entity_set:
        position_dictionary[entity] = index
        index += 1
    return position_dictionary


def build_distance_matrix(distances_pairs, side, position_dictionary):
    distance_matrix = [[1 for _ in range(side)] for _ in range(side)]
    for pair in distances_pairs:
        row = position_dictionary[pair[0]]
        column = position_dictionary[pair[1]]
        distance_matrix[row][column] = pair[2]
    return distance_matrix


def build_inverse_dictionary(position_dictionary):
    inverse_dictionary = {}
    for key, value in position_dictionary.items():
        inverse_dictionary[value] = key
    return inverse_dictionary


def perform_agglomerative_clustering(entity_set, semantic_distances, distance_threshold):
    position_dictionary = build_position_dictionary(entity_set)
    side = len(entity_set)
    distance_matrix = build_distance_matrix(semantic_distances, side, position_dictionary)
    model = AgglomerativeClustering(affinity='precomputed', n_clusters=None,
                                    distance_threshold=distance_threshold, linkage="single")
    results = model.fit(distance_matrix)
    print(max(results.labels_))
    # plot_clustering_labels(results.labels_, side)
    return structure_agglomerative_clustering_results(results, build_inverse_dictionary(position_dictionary))


def structure_agglomerative_clustering_results(results, inverse_position_dictionary):
    clusters = {}
    for index, label in enumerate(results.labels_):
        clusters[label] = clusters.get(label, [])
        clusters[label].append(inverse_position_dictionary[index])  # add name or title if you want that
    # pprint(clusters)
    return clusters


def plot_clustering_labels(labels, side):
    plt.scatter(range(side), range(side), c=labels, cmap='rainbow')
    plt.show()


def build_degree_dictionary(rankings):
    degree_dictionary = {}
    for info in rankings:
        degree_dictionary[info[0]] = info[1]
    return degree_dictionary


def sort_clusters_by_size(clusters, not_used):
    return {k: v for k, v in sorted(clusters.items(), key=lambda item: len(item[1]), reverse=True)}


def sort_clusters_by_degree(clusters, get_rankings_function):
    rankings = get_rankings_function()
    degree_dictionary = build_degree_dictionary(rankings)
    return {k: v for k, v in sorted(clusters.items(), key=lambda item: np.mean([degree_dictionary.get(node, 0)
                                                                               for node in item[1]]), reverse=True)}
