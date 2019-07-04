from rb.core.lang import Lang
from rb.core.document import Document

from rb.docs_processing.author import Author

from typing import List

import datetime


class Article:

    def __init__(self, title: str, abstract: str, source:str, authors: List[str], date: str, lang: Lang, graph: "Graph"):
        self.title = title
        self.abstract = abstract
        self.author_list = []
        
        for author_name in authors:
            if author_name:
                self.author_list.append(author_name)
                if not author_name in graph.authors_dict:
                    author = Author(author_name)
                    graph.authors_dict[author_name] = author
                    author.articles.add(self)
                else:
                    graph.authors_dict[author_name].articles.add(self)
        
        self.source = source
        self.date = datetime.datetime.strptime(date, "%d-%m-%Y")
        self.document = Document(lang, abstract)
        graph.articles_set.add(self)

    
    def __eq__(self, other):
        if isinstance(other, Article):
            return self.title == other.title
        return NotImplemented


    def __hash__(self):
        return hash(tuple(self.title))