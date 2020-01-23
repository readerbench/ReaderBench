from rb.core.lang import Lang
from rb.core.document import Document

from rb.docs_processing.author import Author

from typing import List

import datetime


class Article:

    def __init__(self, title: str, abstract: str, source: str, authors: List[str], date: str, lang: Lang):
        self.title = title
        self.abstract = abstract

        self.author_list = []
        self.extract_authors(authors)
        
        self.source = source
        self.date = datetime.datetime.strptime(date, "%d-%m-%Y")
        self.document = Document(lang, abstract)

    def extract_authors(self, authors):
        for author_name in authors:
            if not author_name:
                continue
            self.author_list.append(author_name.strip())

    def add_this_article_to_its_authors(self, graph):
        for author_name in self.author_list:
            self.add_this_article_to_author(author_name, graph)

    def add_this_article_to_author(self, author_name, graph):
        if author_name not in graph.authors_dict:
            author = Author(author_name)
            author.articles.add(self)
            graph.authors_dict[author_name] = author
        else:
            graph.authors_dict[author_name].articles.add(self)

    def add_this_article_to_the_graph(self, graph):
        graph.articles_set.add(self)

    def __eq__(self, other):
        if isinstance(other, Article):
            return self.title == other.title
        return NotImplemented

    def __hash__(self):
        return hash(tuple(self.title))


def create_article_and_add_it_to_its_authors_and_graph(title: str, abstract: str, source: str,
                                                       authors: List[str], date: str, lang: Lang, graph):
    article = Article(title, abstract, source, authors, date, lang)
    article.add_this_article_to_its_authors(graph)
    article.add_this_article_to_the_graph(graph)

    return article