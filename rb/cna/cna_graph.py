from typing import Callable, List, Tuple, Union

import networkx as nx
from rb.cna.edge_type import EdgeType
from rb.cna.overlap_type import OverlapType
from rb.core.document import Document
from rb.core.pos import POS
from rb.core.text_element import TextElement
from rb.core.word import Word
from rb.similarity.vector_model import VectorModel


class CnaGraph:
    def __init__(self, doc: Document, models: List[VectorModel]):
        self.graph = nx.MultiGraph()
        self.add_element(doc)
        self.models = models
        levels = dict()
        for n in self.graph.nodes:
            if not n.is_word():
                if n.depth in levels:
                    levels[n.depth].append(n)
                else:
                    levels[n.depth] = [n]
        
        for depth, elements in levels.items():
            self.add_lexical_links(elements, lambda w: w.pos in {POS.ADJ, POS.ADV, POS.NOUN, POS.VERB}, OverlapType.CONTENT_OVERLAP)
            self.add_lexical_links(elements, lambda w: w.pos in {POS.NOUN, POS.VERB}, OverlapType.TOPIC_OVERLAP)
            self.add_lexical_links(elements, lambda w: w.pos in {POS.NOUN, POS.PRON}, OverlapType.ARGUMENT_OVERLAP)
            self.add_semantic_links(elements)

        doc.cna_graph = self
        
    def add_element(self, element: TextElement):
        self.graph.add_node(element)
        if not element.is_sentence():
            for child in element.components:
                self.add_element(child)
                self.graph.add_edge(child, element, type=EdgeType.PART_OF)
            self.graph.add_edges_from(zip(element.components[:-1], element.components[1:]), type=EdgeType.ADJACENT)
    
    def add_semantic_links(self, elements: List[TextElement]):
        for i, a in enumerate(elements[:-1]):
            for b in elements[i+1:]:
                for model in self.models:
                    sim = model.similarity(a, b)
                    self.graph.add_edge(a, b, type=EdgeType.SEMANTIC, model=model, value=sim)
                    self.graph.add_edge(b, a, type=EdgeType.SEMANTIC, model=model, value=sim)
        
    def add_lexical_links(self, elements: List[TextElement], test: Callable[[Word], bool], link_type: OverlapType):
        for i, a in enumerate(elements[:-1]):
            for b in elements[i+1:]:
                words_a = {word.lemma for word in a.get_words() if test(word)}
                words_b = {word.lemma for word in b.get_words() if test(word)}
                weight = len(words_a & words_b) / (1e-5 + len(words_a | words_b))
                self.graph.add_edge(a, b, type=EdgeType.LEXICAL_OVERLAP, model=link_type, value=weight)
                self.graph.add_edge(b, a, type=EdgeType.LEXICAL_OVERLAP, model=link_type, value=weight)
                    

    def edges(self, 
            node: Union[TextElement, Tuple[TextElement, TextElement]], 
            edge_type: EdgeType = None, 
            vector_model: Union[VectorModel, OverlapType] = None) -> List[Tuple[TextElement, TextElement, float]]:
        return [(a, b, data["value"] if "value" in data else 0)
            for a, b, data in self.graph.edges(node, data=True) 
            if (edge_type is None or data["type"] is edge_type) and 
               (vector_model is None or data["model"] is vector_model)
        ]
