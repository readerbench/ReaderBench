from itertools import combinations
import networkx as nx
import pandas as pd
from tabulate import tabulate


class CoAuthorship:

    def __init__(self, graph: "Graph"):
        self.co_authorship_dict = {}
        for article in graph.articles_set:
            self.create_co_authorship_for_article_authors(article, graph)

        self.NO_OF_AUTHORS_TO_PRINT = 1000

    def create_co_authorship_for_article_authors(self, article, graph):
        for (author1_name, author2_name) in list(combinations(article.author_list, r=2)):
            self.update_authors_co_authorship_count(author1_name, author2_name, graph)

    def update_authors_co_authorship_count(self, author1_name, author2_name, graph):
        author1 = graph.authors_dict[author1_name]
        author2 = graph.authors_dict[author2_name]
        common_articles_number = self.co_authorship_dict.get((author1, author2), 0)
        self.co_authorship_dict[(author1, author2)] = common_articles_number + 1

    def build_co_authorship_degree_graph(self):
        co_authorship_degree_graph = nx.Graph()
        for (author1, author2), co_authorship_distance in self.co_authorship_dict.items():
            co_authorship_degree_graph.add_node(author1)
            co_authorship_degree_graph.add_node(author2)
            co_authorship_degree_graph.add_edge(author1, author2, weight=co_authorship_distance)
        return co_authorship_degree_graph

    def get_rankings(self):
        co_authorship_degree_graph = self.build_co_authorship_degree_graph()
        co_authorship_degrees = dict(nx.degree(co_authorship_degree_graph, weight='weight'))
        return build_rankings_dictionary(co_authorship_degrees)

    def print_results(self):
        rankings_dictionary = self.get_rankings()
        co_authorship_degrees_df = pd.DataFrame.from_dict(rankings_dictionary)
        sorted_df = co_authorship_degrees_df.sort_values(by=['Degree'], ascending=False).head(
            self.NO_OF_AUTHORS_TO_PRINT)[["AuthorName", "NoOfArticles", "Degree"]]
        print(tabulate(sorted_df, headers='keys', tablefmt='psql', showindex=False))


def build_rankings_dictionary(co_authorship_degrees):
    rankings_dictionary = {"AuthorName": [], "NoOfArticles": [], "Degree": []}
    for author, degree in co_authorship_degrees.items():
        rankings_dictionary["AuthorName"].append(author.name)
        rankings_dictionary["NoOfArticles"].append(len(author.articles))
        rankings_dictionary["Degree"].append(degree)
    return rankings_dictionary

