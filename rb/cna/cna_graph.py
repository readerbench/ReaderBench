from typing import List, Tuple, Union

import networkx as nx
from rb.cna.edge_type import EdgeType
from rb.core.document import Document
from rb.core.text_element import TextElement
from rb.similarity.vector_model import VectorModel


class CnaGraph:
    def __init__(self, doc: Document, models: List[VectorModel]):
        self.graph = nx.MultiGraph()
        self.add_element(doc)
        self.models = models
        self.add_lexical_links()
        self.add_semantic_links()
        doc.cna_graph = self
        
    def add_element(self, element: TextElement):
        self.graph.add_node(element)
        if not element.is_sentence():
            for child in element.components:
                self.add_element(child)
                self.graph.add_edge(child, element, type=EdgeType.PART_OF)
            self.graph.add_edges_from(zip(element.components[:-1], element.components[1:]), type=EdgeType.ADJACENT)
    
    def add_semantic_links(self):
        for i, a in enumerate(list(self.graph.nodes)[:-1]):
            for b in list(self.graph.nodes)[i+1:]:
                for model in self.models:
                    sim = model.similarity(a, b)
                    self.graph.add_edge(a, b, type=EdgeType.SEMANTIC, model=model, value=sim)
                    self.graph.add_edge(b, a, type=EdgeType.SEMANTIC, model=model, value=sim)

    def add_lexical_links(self):
        non_words: List[TextElement] = [node for node in self.graph.nodes if not node.is_word()]
        for i, a in enumerate(non_words[:-1]):
            for b in non_words[i+1:]:
                words_a = {word.lemma for word in a.get_words()}
                words_b = {word.lemma for word in b.get_words()}
                weight = len(words_a & words_b) / len(words_a | words_b)
                self.graph.add_edge(a, b, type=EdgeType.LEXICAL, value=weight)
                self.graph.add_edge(b, a, type=EdgeType.LEXICAL, value=weight)
                    


    def edges(self, 
            node: Union[TextElement, Tuple[TextElement, TextElement]], 
            edge_type: EdgeType = None, 
            vector_model: VectorModel = None) -> List[Tuple[TextElement, TextElement, float]]:
        return [(a, b, data["value"] if "value" in data else 0)
            for a, b, data in self.graph.edges(node, data=True) 
            if (edge_type is None or data["type"] is edge_type) and 
               (vector_model is None or data["model"] is vector_model)
        ]
