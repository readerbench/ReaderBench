from typing import Dict, List
from rb.docs_processing.author import Author
from rb.docs_processing.article import Article
from rb.similarity.vector_model import VectorModel


class Graph:

    def __init__(self):
        self.authors_dict: Dict[str, Author] = {}
        self.articles_set = set()
        self.adjacent_list = {}


    def extract_authors_dict(self) -> None:
        self.authors_set = set([x for _, x in self.authors_dict.items()])


    def get_maximum_coupling(self, author1_details, author2_details, sim_dict):
        from munkres import Munkres, DISALLOWED, print_matrix
        author1_articles = author1_details[1]
        author2_articles = author2_details[1]

        m = [[DISALLOWED for x in range(len(author2_articles))] for y in range(len(author1_articles))]

        for i1, art1 in enumerate(author1_articles):
            for i2, art2 in enumerate(author2_articles):
                if (art1, art2) in sim_dict:
                    if sim_dict[(art1, art2)] > 0.3:
                        m[i1][i2] = (1 - sim_dict[(art1, art2)]) * 100
                elif (art2, art1) in sim_dict:
                    if sim_dict[(art2, art1)] > 0.3:
                        m[i1][i2] = (1 - sim_dict[(art2, art1)]) * 100


        mumu = Munkres()
        try:
            indexes = mumu.compute(m)
        except:
            return 1.0
        total = 0
        for row, column in indexes:
            value = 1 - (m[row][column] / 100)
            total += value
        
        if total > 0:
            return total
        return 1.0


    def compute_semantic_distance_between_articles(self, article1: Article, article2: Article, semantic_models: List[VectorModel]) -> float:
        if article1 == article2:
            return 0.0

        similarity = 0.0
        for model in semantic_models:
            similarity += model.similarity(article1.document, article2.document)

        if similarity != 0:
            return 1 - similarity / len(semantic_models)
        return 1.0


    def compute_semantic_distance_between_authors(self, author1: Author, author2: Author, semantic_models: List[VectorModel]) -> float:
        if author1 == author2:
            return 0.0
        
        sim_dict = {}
        for article1 in author1.articles:
            for article2 in author2.articles:
                sim_dict[(article1.title, article2.title)] = 1 - self.compute_semantic_distance_between_articles(article1, article2, semantic_models)
        
        if len(sim_dict) == 1:
            v = next(iter(sim_dict.values()))
            return 1 - v if v > 0.3 else 1.0

        a1 = (author1.name, list(set([x[0] for x in sim_dict.keys()])))
        a2 = (author2.name, list(set([x[1] for x in sim_dict.keys()])))
        if sim_dict:
            return self.get_maximum_coupling(a1, a2, sim_dict)
        
        return 1.0


    def compute_semantic_distance_between_author_and_article(self, author: Author, article: Article, semantic_models: List[VectorModel]) -> float:
        if article in author.articles:
            return 0.0
        
        distance = 0.0
        for aa in author.articles:
            distance += self.compute_semantic_distance_between_articles(aa, article, semantic_models)

        if distance > 0:
            return distance / len(author.articles)
        
        return 1.0


    def add_to_adjacent_list(self, entity1, entity2, dist: float, link_type: str) -> None:
        if entity1 in self.adjacent_list:
            self.adjacent_list[entity1].append((entity2, dist, link_type))
        else:
            self.adjacent_list[entity1] = [(entity2, dist, link_type)]

        if entity2 in self.adjacent_list:
            self.adjacent_list[entity2].append((entity1, dist, link_type))
        else:
            self.adjacent_list[entity2] = [(entity1, dist, link_type)]


    def build_edges(self, semantic_models: List[VectorModel]) -> None:
        articles_list = list(self.articles_set)
        authors_list = list(self.authors_set)

        max_distance = 0.8

        lart = len(articles_list)
        laut = len(authors_list)

        for i in range(lart - 1):
            for j in range(i + 1, lart):
                dist = self.compute_semantic_distance_between_articles(articles_list[i], articles_list[j], semantic_models)
                if dist < max_distance:
                    self.add_to_adjacent_list(articles_list[i], articles_list[j], dist, 'art-art')

        for i in range(laut - 1):
            for j in range(i + 1, laut):
                dist = self.compute_semantic_distance_between_authors(authors_list[i], authors_list[j], semantic_models)
                if dist < max_distance:
                    self.add_to_adjacent_list(authors_list[i], authors_list[j], dist, 'aut-aut')

        for i in range(lart):
            for j in range(laut):
                dist = self.compute_semantic_distance_between_author_and_article(authors_list[j], articles_list[i], semantic_models)
                if dist < max_distance:
                    self.add_to_adjacent_list(authors_list[j], articles_list[i], dist, 'aut-art')

    
    def get_authors_by_type_degree(self, max_ = None):
        authors = [(author, sum(n for _, n, t in self.adjacent_list[author] if t == 'aut-aut') / len(author.articles)) for author in self.authors_set]
        if max_:
            return sorted(authors, key=lambda x: x[1])[:max_]
        return sorted(authors, key=lambda x: x[1])


    def get_articles_by_type_degree(self, max_ = None):
        articles = [(article, sum(n for _, n, t in self.adjacent_list[article] if t == 'art-art')) for article in self.articles_set]
        if max_:
            return sorted(articles, key=lambda x: x[1])[:max_]
        return sorted(articles, key=lambda x: x[1])

    
    def get_distances_between_articles(self):
        pairs = []
        used_articles = set()
        for article in self.articles_set:
            for neigh, score, edge_type in self.adjacent_list[article]:
                if edge_type == 'art-art' and neigh not in used_articles:
                    pairs.append((article, neigh, score))
            used_articles.add(article)

        return pairs

