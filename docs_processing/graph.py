from typing import Dict, List
from munkres import Munkres, DISALLOWED
from rb.docs_processing.author import Author
from rb.docs_processing.article import Article
from rb.similarity.vector_model import VectorModel
from rb.similarity.transformers_encoder import TransformersEncoder

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


class Graph:
    EDGE_THRESHOLD = 0.064
    BIG_VALUE = 10000
    NO_EDGE_PENALTY = 100
    # EDGE_THRESHOLD = 1.01

    def __init__(self, semantic_models: List[VectorModel]):
        self.authors_dict: Dict[str, Author] = {}
        self.articles_set = set()
        self.authors_set = set()
        self.adjacent_list = {}
        self.authors_ranking_graph = {}
        self.articles_distances_dict = {}
        self.semantic_models = semantic_models
        self.articles_mean = 0.0
        self.articles_std = 0.0
        self.authors_mean = 0.0
        self.authors_std = 0.0

    def extract_authors_dict(self) -> None:
        self.authors_set = set([x for _, x in self.authors_dict.items()])

    def add_to_adjacent_list(self, entity1, entity2, dist: float, link_type: str, direction: str = "bidirectional") -> None:
        if entity1 in self.adjacent_list:
            self.adjacent_list[entity1].append((entity2, dist, link_type))
        else:
            self.adjacent_list[entity1] = [(entity2, dist, link_type)]

        if direction == "bidirectional":
            if entity2 in self.adjacent_list:
                self.adjacent_list[entity2].append((entity1, dist, link_type))
            else:
                self.adjacent_list[entity2] = [(entity1, dist, link_type)]

    def build_edges_and_construct_author_rankings_graph(self, authors_relation_type="bidirectional") -> None:
        articles_list = list(self.articles_set)
        authors_list = list(self.authors_set)

        self.build_edges_between_articles_and_save_the_distances(articles_list)
        if not authors_relation_type:
            return
        
        self.build_edges_between_articles_and_authors(articles_list, authors_list)
        authors_semantic_similarities_dictionary = self.compute_semantic_similarities_between_authors(authors_list, authors_relation_type)
        self.build_authors_ranking_graph(authors_semantic_similarities_dictionary, authors_relation_type=authors_relation_type)
        self.build_edges_between_authors(authors_semantic_similarities_dictionary, authors_relation_type=authors_relation_type)

    def build_edges_between_articles_and_authors(self, articles_list, authors_list):
        for i in range(len(articles_list)):
            for j in range(len(authors_list)):
                dist = compute_semantic_distance_between_author_and_article(authors_list[j], articles_list[i],
                                                                            self.semantic_models)
                if dist < Graph.EDGE_THRESHOLD:
                    self.add_to_adjacent_list(authors_list[j], articles_list[i], dist, 'aut-art')

    def build_edges_between_articles_and_save_the_distances(self, articles_list):
        counter = 0
        articles_to_distances_dict = self.build_article_pairs_to_distance_dict(articles_list)
        threshold, self.articles_mean, self.articles_std = \
            compute_edge_threshold_mean_and_std(articles_to_distances_dict)

        for articles_pair, distance in articles_to_distances_dict.items():
            article1 = articles_pair[0]
            article2 = articles_pair[1]
            if distance < threshold:
                counter += 1
                # print(counter)
                self.add_to_adjacent_list(article1, article2, distance, 'art-art')
                self.articles_distances_dict[(article1, article2)] = distance
            else:
                self.articles_distances_dict[(article1, article2)] = 1.0

    def build_article_pairs_to_distance_dict(self, articles_list):
        articles_to_distances_dict = {}
        for i in range(len(articles_list) - 1):
            for j in range(i + 1, len(articles_list)):
                article1 = articles_list[i]
                article2 = articles_list[j]
                distance = compute_distance_between_articles(article1, article2, self.semantic_models)
                articles_to_distances_dict[(article1, article2)] = distance
        return articles_to_distances_dict

    def build_edges_between_authors(self, authors_semantic_similarities_dictionary, authors_relation_type):
        if authors_relation_type == "bidirectional":
            similarities = authors_semantic_similarities_dictionary.values()
            maximum_value, minimal_value, self.authors_mean, self.authors_std = extract_min_max_mean_std(similarities)
            threshold = self.authors_mean - self.authors_std
            for authors_pair, similarity in authors_semantic_similarities_dictionary.items():
                scaled_distance = 1 - (similarity - minimal_value) / (maximum_value - minimal_value)
                if scaled_distance < threshold:
                    self.add_to_adjacent_list(authors_pair[0], authors_pair[1], scaled_distance, 'aut-aut')
        elif authors_relation_type == 'unidirectional':
            distances = [1 - x for x in authors_semantic_similarities_dictionary.values()]
            self.authors_mean = np.mean(distances)
            self.authors_std = np.std(distances)
            for authors_pair, similarity in authors_semantic_similarities_dictionary.items():
                self.add_to_adjacent_list(authors_pair[0], authors_pair[1], 1 - similarity, 'aut-aut', direction='unidirectional')
        else:
            raise Exception("Unknown authors relation type")

    def build_authors_ranking_graph(self, authors_semantic_similarities_dictionary, authors_relation_type='bidirectional'):
        for authors_pair, similarity in authors_semantic_similarities_dictionary.items():
            author1 = authors_pair[0]
            author2 = authors_pair[1]
            if authors_relation_type == 'bidirectional':
                self.authors_ranking_graph[author1] = self.authors_ranking_graph.get(author1, [])
                self.authors_ranking_graph[author1].append((author2, similarity))
                self.authors_ranking_graph[author2] = self.authors_ranking_graph.get(author2, [])
                self.authors_ranking_graph[author2].append((author1, similarity))
            elif authors_relation_type == 'unidirectional':
                self.authors_ranking_graph[author1] = self.authors_ranking_graph.get(author1, [])
                self.authors_ranking_graph[author1].append((author2, similarity))
            else:
                raise Exception('Invalid authors_relation_type')

    def compute_semantic_similarities_between_authors(self, authors_list, authors_relation_type="bidirectional"):
        authors_semantic_similarities_dictionary = {}
        for i in range(len(authors_list) - 1):
            for j in range(i + 1, len(authors_list)):
                author1 = authors_list[i]
                author2 = authors_list[j]
                if authors_relation_type == "bidirectional":
                    similarity = self.compute_semantic_similarity_between_authors(author1, author2)
                    authors_semantic_similarities_dictionary[(author1, author2)] = similarity
                elif authors_relation_type == "unidirectional":
                    print(f"Computing semantic similarity between {author1} with {len(author1.articles)} and {author2} with {len(author2.articles)}")
                    similarity = self.compute_semantic_similarity_between_author1_and_author2(author1, author2)
                    authors_semantic_similarities_dictionary[(author1, author2)] = similarity
                    similarity = self.compute_semantic_similarity_between_author1_and_author2(author2, author1)
                    authors_semantic_similarities_dictionary[(author2, author1)] = similarity
                else:
                    raise Exception("Invalid authors_relation_type")
                # if dist < max_distance:
                #     self.add_to_adjacent_list(authors_list[i], authors_list[j], dist, 'aut-aut')
        return authors_semantic_similarities_dictionary

    def get_authors_by_type_degree(self, max_=None):
        authors = [(author, sum(n for _, n in self.authors_ranking_graph[author])) for author in self.authors_set]
        if max_:
            return sorted(authors, key=lambda x: x[1], reverse=True)[:max_]
        return sorted(authors, key=lambda x: x[1], reverse=True)
    
    
    def get_authors_by_out_type_degree(self, max_=None):
        return self.get_authors_by_type_degree(max_)
    
    
    def get_authors_by_in_type_degree(self, max_=None):
        inverse_dict = {}
        for author in self.authors_set:
            for neighbour, similarity in self.authors_ranking_graph[author]:
                inverse_dict[neighbour] = inverse_dict.get(neighbour, 0) + similarity
        authors = [(author, inverse_dict[author]) for author in self.authors_set]
        if max_:
            return sorted(authors, key=lambda x: x[1], reverse=True)[:max_]
        return sorted(authors, key=lambda x: x[1], reverse=True)
    

    def get_articles_by_type_degree(self, max_=None):
        articles = [(article, sum(1 - n for _, n, t in self.adjacent_list[article] if t == 'art-art'),
                     len([t for _, _, t in self.adjacent_list[article] if t == 'art-art'])) for article in
                    self.articles_set if article in self.adjacent_list]
        if max_:
            return sorted(articles, key=lambda x: x[1], reverse=True)[:max_]
        return sorted(articles, key=lambda x: x[1], reverse=True)

    def get_similarities_between_articles(self):
        similarities_dictionary = {}
        for pair in self.get_distances_between_articles():
            similarities_dictionary[(pair[0], pair[1])] = 1 - pair[2]
        return similarities_dictionary

    def get_similarities_between_authors(self):
        similarities_dictionary = {}
        for author, neighbours in self.authors_ranking_graph.items():
            for neighbour in neighbours:
                if (neighbour[0], author) not in similarities_dictionary:
                    similarities_dictionary[(author, neighbour[0])] = neighbour[1]
        return similarities_dictionary

    def get_distances_between_articles(self):
        return self.get_distances_between_nodes_of_same_type("art-art", self.articles_set)

    def get_distances_between_authors(self):
        return self.get_distances_between_nodes_of_same_type("aut-aut", self.authors_set)

    def get_distances_between_nodes_of_same_type(self, node_type, node_set):
        pairs = []
        used_nodes = set()
        for node in node_set:
            if node in self.adjacent_list:
                for neigh, score, edge_type in self.adjacent_list[node]:
                    if edge_type == node_type and neigh not in used_nodes:
                        pairs.append((node, neigh, score))
                used_nodes.add(node)
        return pairs

    def plot_histogram_of_articles_distance(self):
        distances = [pair[2] for pair in self.get_distances_between_articles()]
        print(np.mean(distances))
        print(np.std(distances))
        plt.hist(distances, color="blue", edgecolor="black")
        plt.show()
        
    # only one direction, author1 -> author2
    def compute_semantic_similarity_between_author1_and_author2(self, author1, author2):
        if author1 == author2:
            return len(author1.articles)
        
        if len(author1.articles) >= len(author2.articles):
            return self.compute_semantic_similarity_between_authors(author1, author2)
        
        to_be_excluded_author2_articles = set()
        similarities = []
        while True:
            distance_dict = self.build_articles_distance_dictionary(author1, author2, to_be_excluded_author2_articles)
            if len(distance_dict) == 1:
                similarities.append(1 - next(iter(distance_dict.values())))
                return np.mean(similarities)
            
            if not distance_dict:
                return np.mean(similarities)
            
            author1_name_articles_tuple = (author1.name, list(set([x[0] for x in distance_dict.keys()])))
            author2_name_articles_tuple = (author2.name, list(set([x[1] for x in distance_dict.keys()])))
            
            articles_distances_matrix = build_distances_between_articles_matrix(
                author1_name_articles_tuple[1],
                author2_name_articles_tuple[1],
                distance_dict)
            
            # check if no edges
            are_edges = False
            for e in articles_distances_matrix:
                for i in e:
                    if i != Graph.BIG_VALUE:
                        are_edges = True
                        break
                if are_edges:
                    break
            if not are_edges:
                similarities += [0] * min(len(author1.articles), len(author2.articles) - len(to_be_excluded_author2_articles))
                return np.mean(similarities)
            
            munkres_client = Munkres()
            try:
                indexes = munkres_client.compute(articles_distances_matrix)
            except:
                if similarities:
                    return max(np.mean(similarities) - 0.001, 0) # epsilon in case of an error
                return 0.0  # minimum similarity possible
            
            for row, column in indexes:
                if articles_distances_matrix[row][column] == Graph.BIG_VALUE:
                    continue
                value = 1 - articles_distances_matrix[row][column] / 100
                similarities.append(value)
                to_be_excluded_author2_articles.add(author2_name_articles_tuple[1][column])


    def compute_semantic_similarity_between_authors(self, author1, author2):
        if author1 == author2:
            return len(author1.articles)

        distance_dict = self.build_articles_distance_dictionary(author1, author2)
        # (distance_dict)
        if len(distance_dict) == 1:
            return 1 - next(iter(distance_dict.values()))

        if distance_dict:
            a1 = (author1.name, list(set([x[0] for x in distance_dict.keys()])))
            a2 = (author2.name, list(set([x[1] for x in distance_dict.keys()])))
            return compute_maximum_coupling_of_maximum_similarity(a1, a2, distance_dict)

        return 0.0
    

    def build_articles_distance_dictionary(self, author1, author2, to_be_excluded_author2_articles=set()):
        distance_dict = {}
        for article1 in author1.articles:
            for article2 in author2.articles:
                if article2.title in to_be_excluded_author2_articles:
                    continue
                if article1 == article2:
                    distance_dict[(article1.title, article2.title)] = 1 - (1 / len(article1.author_list))
                else:
                    distance_dict[(article1.title, article2.title)] = \
                        self.articles_distances_dict.get((article1, article2), 0) + \
                        self.articles_distances_dict.get((article2, article1), 0)
        return distance_dict


def compute_maximum_coupling_of_maximum_similarity(author1_name_articles_tuple, author2_name_articles_tuple,
                                                   distance_dictionary):
    articles_distances_matrix = build_distances_between_articles_matrix(
        author1_name_articles_tuple[1],
        author2_name_articles_tuple[1],
        distance_dictionary)
    # print(articles_distances_matrix)
    munkres_client = Munkres()
    try:
        indexes = munkres_client.compute(articles_distances_matrix)
    except:
        return 0.0  # minimum similarity possible
    return compute_semantic_similarity_coupling_value(articles_distances_matrix, indexes)


def compute_semantic_similarity_coupling_value(articles_distances_matrix, indexes):
    total = 0
    for row, column in indexes:
        if articles_distances_matrix[row][column] == Graph.BIG_VALUE:
            continue
        value = 1 - articles_distances_matrix[row][column] / 100
        total += value
    return total


def clear_matrix_of_only_disallowed(distances_matrix, only_disallowed_lines_index):
    only_disallowed_lines_index.reverse()
    for index in only_disallowed_lines_index:
        del distances_matrix[index]

    if not distances_matrix:
        return

    only_disallowed_columns_index = []
    for j in range(len(distances_matrix[0])):
        was_only_disallowed = True
        for i in range(len(distances_matrix)):
            if distances_matrix[i][j] != DISALLOWED:
                was_only_disallowed = False
                break
        if was_only_disallowed:
            only_disallowed_columns_index.append(j)

    only_disallowed_columns_index.reverse()

    for line in distances_matrix:
        for index in only_disallowed_columns_index:
            del line[index]


def build_distances_between_articles_matrix(author1_articles, author2_articles,
                                            distance_dictionary):
    distances_matrix = [[Graph.BIG_VALUE for x in range(len(author2_articles))] for y in
                        range(len(author1_articles))]
    for i1, art1 in enumerate(author1_articles):
        for i2, art2 in enumerate(author2_articles):
            if (art1, art2) in distance_dictionary and distance_dictionary[(art1, art2)] < Graph.EDGE_THRESHOLD:
                distances_matrix[i1][i2] = distance_dictionary[(art1, art2)] * 100
            elif (art2, art1) in distance_dictionary and distance_dictionary[(art2, art1)] < Graph.EDGE_THRESHOLD:
                distances_matrix[i1][i2] = distance_dictionary[(art2, art1)] * 100

    return distances_matrix


def compute_distance_between_articles(article1: Article, article2: Article,
                                      semantic_models: List[VectorModel]) -> float:
    if article1 == article2:
        return 0.0

    similarity = 0.0
    for model in semantic_models:
        if type(model) is TransformersEncoder:
            similarity += model.similarity(article1.document.vectors[model], article2.document.vectors[model])
        else:
            similarity += model.similarity(article1.document, article2.document)

    if similarity != 0:
        distance = 1 - similarity / len(semantic_models)
        return distance if distance >= 0 else 0
    return 1.0


def compute_semantic_distance_between_author_and_article(author: Author, article: Article,
                                                         semantic_models: List[VectorModel]) -> float:
    if article in author.articles:
        return 0.0

    distance = 0.0
    for aa in author.articles:
        distance += compute_distance_between_articles(aa, article, semantic_models)

    if distance > 0:
        return distance / len(author.articles)

    return 1.0


def extract_min_max_mean_std(similarities):
    minimal_value = min(similarities)
    maximum_value = max(similarities)
    scaled_distances = [1 - (similarity - minimal_value) / (maximum_value - minimal_value)
                        for similarity in similarities]
    mean = np.mean(scaled_distances)
    std = np.std(scaled_distances)
    return maximum_value, minimal_value, mean, std


def compute_edge_threshold_mean_and_std(articles_to_distances_dict):
    values = articles_to_distances_dict.values()
    mean = np.mean(list(values))
    std = np.std(list(values))
    # print(f"mean: {mean}, std: {std}")
    threshold = mean - std
    return threshold, mean, std
