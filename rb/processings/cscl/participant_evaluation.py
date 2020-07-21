from typing import Dict, List

import networkx as nx
from rb.cna.cna_graph import CnaGraph
from rb.complexity.complexity_index import compute_indices
from rb.core.block import Block
from rb.core.cscl.cna_indices_enum import CNAIndices
from rb.core.cscl.contribution import Contribution
from rb.core.cscl.conversation import Conversation
from rb.core.lang import Lang
from rb.core.pos import POS
from rb.core.text_element import TextElement
from rb.utils.rblogger import Logger


def get_block_importance(graph: nx.DiGraph, a: Block, b: Block) -> float:
	return graph.edges[a, b]["weight"] if (a, b) in graph.edges else 0.


def evaluate_interaction(conversation: Conversation):
	cna_graph = conversation.container.graph if conversation.container is not None else conversation.graph
	
	participants = conversation.get_participants()
	contributions = conversation.get_contributions()
	conversation.init_scores()

	if (len(participants) == 0):
		return

	for i, contribution1 in enumerate(contributions):
		p1 = contribution1.get_participant().get_id()

		for contribution2 in contributions[:i]:
			weight = get_block_importance(cna_graph.filtered_graph, contribution1, contribution2)
			if weight > 0:
				p2 = contribution2.get_participant().get_id()
				conversation.update_score(p1, p2, weight)


def evaluate_involvement(conversation: Conversation):
	cna_graph = conversation.container.graph if conversation.container is not None else conversation.graph
	participants = conversation.get_participants()

	if (len(participants) == 0):
		return

	for contribution in conversation.get_contributions():
		p = contribution.get_participant()

		current_value = p.get_index(CNAIndices.SCORE)
		p.set_index(CNAIndices.SCORE, current_value + cna_graph.importance[contribution])

		current_value = p.get_index(CNAIndices.SOCIAL_KB)
		parent_contribution = contribution.get_parent()

		if parent_contribution != None:
			current_value += get_block_importance(cna_graph.filtered_graph, contribution, parent_contribution)
			p.set_index(CNAIndices.SOCIAL_KB, current_value)

def evaluate_textual_complexity(conversation: Conversation):
	participants = conversation.get_participants()
	
	for p in participants:
		contributions = conversation.get_participant_contributions(p.get_id())
		p.set_eligible_contributions(contributions)

		Conversation.create_participant_conversation(conversation.lang, p)

		own_conversation = p.own_conversation

		print('Begin computing textual complexity')
		compute_indices(doc=own_conversation)
		print('Finished computing textual complexity')

		for key, value in own_conversation.indices.items():
			p.set_textual_index(repr(key), value)


def perform_sna(conversation: Conversation, needs_anonymization: bool):
	participants = conversation.get_participants()

	for i, p1 in enumerate(participants):
		for j, p2 in enumerate(participants):
			if i != j:
				current_value = p1.get_index(CNAIndices.OUTDEGREE)
				p1.set_index(CNAIndices.OUTDEGREE, current_value +
											conversation.get_score(p1.get_id(), p2.get_id()))

				current_value = p2.get_index(CNAIndices.INDEGREE)
				p2.set_index(CNAIndices.INDEGREE, current_value +
											conversation.get_score(p1.get_id(), p2.get_id()))
			else:
				current_value = p1.get_index(CNAIndices.OUTDEGREE)
				p1.set_index(CNAIndices.OUTDEGREE, current_value +
											conversation.get_score(p1.get_id(), p1.get_id()))
