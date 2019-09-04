import networkx as nx
from rb.cna.edge_type import EdgeType
from rb.core.document import Document
from rb.core.text_element import TextElement
from rb.similarity.vector_model import VectorModel


class CnaGraph:
    def __init__(self, doc: Document, model: VectorModel):
        self.graph = nx.MultiGraph()
        self.add_element(doc)
        self.model = model
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
                sim = self.model.similarity(a, b)
                if sim > 0.3:
                    self.graph.add_edge(a, b, type=EdgeType.SEMANTIC)
                    self.graph.add_edge(b, a, type=EdgeType.SEMANTIC)