from typing import Callable, Dict, List, Tuple, Union

import networkx as nx
import numpy as np
from rb.cna.edge_type import EdgeType
from rb.cna.overlap_type import OverlapType
from rb.core.block import Block
from rb.core.cscl.community import Community
from rb.core.cscl.contribution import Contribution
from rb.core.cscl.conversation import Conversation
from rb.core.document import Document
from rb.core.lang import Lang
from rb.core.meta_document import MetaDocument
from rb.core.pos import POS
from rb.core.text_element import TextElement
from rb.core.word import Word
from rb.similarity.vector_model import VectorModel


class CnaGraph:
	def __init__(self, docs: Union[TextElement, List[TextElement]], models: List[VectorModel]):
		if isinstance(docs, TextElement):
			docs = [docs]
		self.graph = nx.MultiDiGraph()
		self.models = models
		if all(isinstance(doc, Community) or isinstance(doc, MetaDocument) for doc in docs):
			for doc in docs:
				self.graph.add_node(doc)
				for conv in doc.components:
					nodes = self.add_element(conv)
					self.graph.add_edge(conv, doc, type=EdgeType.PART_OF)
					self.add_links(nodes)
		else:
			for doc in docs:
				self.add_element(doc)
			self.add_links(self.graph.nodes)
		
		self.importance = self.compute_importance()

		self.add_coref_links()
		self.add_explicit_links()
		self.block_importance = self.compute_block_importance()

	def add_links(self, nodes):
		levels = dict()
		for n in nodes:
			if not n.is_word():
				if n.depth in levels:
					levels[n.depth].append(n)
				else:
					levels[n.depth] = [n]
		for depth, elements in levels.items():
			self.add_lexical_links(elements, lambda w: w.pos in [POS.ADJ, POS.ADV, POS.NOUN, POS.VERB], OverlapType.CONTENT_OVERLAP)
			self.add_lexical_links(elements, lambda w: w.pos in [POS.NOUN, POS.VERB], OverlapType.TOPIC_OVERLAP)
			self.add_lexical_links(elements, lambda w: w.pos in [POS.NOUN, POS.PRON], OverlapType.ARGUMENT_OVERLAP)
			self.add_semantic_links(elements)

	def add_element(self, element: TextElement) -> List[TextElement]:
		self.graph.add_node(element)
		result = [element]
		if not element.is_sentence():
			for child in element.components:
				result += self.add_element(child)
				self.graph.add_edge(child, element, type=EdgeType.PART_OF)
			self.graph.add_edges_from(zip(element.components[:-1], element.components[1:]), type=EdgeType.ADJACENT)
		return result

	def add_semantic_links(self, elements: List[TextElement]):
		for i, a in enumerate(elements[:-1]):
			for b in elements[i+1:]:
				for model in self.models:
					sim = model.similarity(a, b)
					self.graph.add_edge(a, b, type=EdgeType.SEMANTIC, model=model, value=sim)
					self.graph.add_edge(b, a, type=EdgeType.SEMANTIC, model=model, value=sim)
		
	def add_lexical_links(self, elements: List[TextElement], test: Callable[[Word], bool], link_type: OverlapType):
		words = {element: {word.lemma for word in element.get_words() if test(word)} for element in elements}
		for i, a in enumerate(elements[:-1]):
			for b in elements[i+1:]:
				words_a = words[a]
				words_b = words[b]
				weight = len(words_a & words_b) / (1e-5 + len(words_a | words_b))
				self.graph.add_edge(a, b, type=EdgeType.LEXICAL_OVERLAP, model=link_type, value=weight)
				self.graph.add_edge(b, a, type=EdgeType.LEXICAL_OVERLAP, model=link_type, value=weight)
					
	def add_coref_links(self):
		for node in self.graph.nodes():
			if isinstance(node, Block):
				if node.has_coref:
					for cluster in node.coref_clusters:
						for mention in cluster.mentions:
							if mention != cluster.main and mention.container != cluster.main.container:
								edge = self.get_edge(mention.container, cluster.main.container, edge_type=EdgeType.COREF)
								if edge is None:
									self.graph.add_edge(mention.container, cluster.main.container, type=EdgeType.COREF, details=[(mention.text, cluster.main.text)])
								else:
									edge["details"].append((mention.text, cluster.main.text))


	def add_explicit_links(self):
		explicit_links = []

		for node in self.graph.nodes:
			if isinstance(node, Contribution):
				parent = node.get_parent()
				explicit_links.append((node, parent))

		for node, parent in explicit_links:
			self.graph.add_edge(node, parent, type=EdgeType.EXPLICIT)

	def is_coref_edge(self, a: Block, b: Block) -> bool:
		for _, x, _ in self.edges(a, edge_type=EdgeType.COREF):
			if x is b:
				return True

		return False

	def is_explicit_edge(self, a: Block, b: Block) -> bool:
		for _, x, _ in self.edges(a, edge_type=EdgeType.EXPLICIT):
			if x is b:
				return True

		return False

	def compute_block_importance(self) -> Dict[Block, Dict[Block, float]]:
		block_links = [
        	(a, b, edge["value"])
        	for a, nbrsdict in self.graph.adjacency()
			if isinstance(a, Block)
			for b, edges in nbrsdict.items()
			if isinstance(b, Block)
			for edge in edges.values()
			if edge["type"] == EdgeType.SEMANTIC
		]

		mean = np.mean([value for _, _, value in block_links])
		stdev = np.std([value for _, _, value in block_links])

		block_importance = dict()

		for a, b, value in block_links:
			if not (a in block_importance):
				block_importance[a] = dict()

			if self.is_coref_edge(a, b) or self.is_coref_edge(b, a):
				block_importance[a][b] = value
			elif self.is_explicit_edge(a, b) or self.is_explicit_edge(b, a):
				block_importance[a][b] = value
			elif value > mean + stdev:
				block_importance[a][b] = value
			else:
				block_importance[a][b] = 0

		return block_importance

	def compute_importance(self) -> Dict[TextElement, float]:
		similarities = [value for _, _, value in self.edges(None, edge_type=EdgeType.SEMANTIC)]
		mean = np.mean(similarities)
		stdev = np.std(similarities)
		importance = {}
		for node in self.graph.nodes:
			importance[node] = sum([value for _, _, value in self.edges(node, edge_type=EdgeType.SEMANTIC) if value > mean + stdev])
		return importance

	def edges(self, 
			node: Union[TextElement, Tuple[TextElement, TextElement]], 
			edge_type: EdgeType = None, 
			vector_model: Union[VectorModel, OverlapType] = None) -> List[Tuple[TextElement, TextElement, float]]:
		return [(a, b, data["value"] if "value" in data else 0)
			for a, b, data in self.graph.edges(node, data=True) 
			if (edge_type is None or data["type"] is edge_type) and 
			   (vector_model is None or data["model"] is vector_model)
		]


	def get_edge(self, a: TextElement, b: TextElement, edge_type: EdgeType) -> Dict:
		edge = self.graph[a][b]
		for data in self.graph[a][b].values():
			if (data["type"] is edge_type):
				return data
		return None
