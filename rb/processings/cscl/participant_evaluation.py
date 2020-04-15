from typing import List, Dict

from rb.core.lang import Lang
from rb.core.text_element import TextElement
from rb.core.cscl.contribution import Contribution
from rb.core.cscl.conversation import Conversation
from rb.cna.cna_graph import CnaGraph
from rb.core.cscl.cscl_indices import CsclIndices

from rb.utils.rblogger import Logger

class ParticipantEvaluation:

	@staticmethod
	def evaluate_interaction(conversation: Conversation):
		cna_graph = conversation.container.graph
		importance = cna_graph.importance
		block_importance = cna_graph.compute_block_importance()

		participants = conversation.get_participants()
		contributions = conversation.get_contributions()
		conversation.init_scores()

		if (len(participants) == 0):
			return

		for i, contribution1 in enumerate(contributions):
			p1 = contribution1.get_participant().get_id()

			conversation.set_score(p1, p1, importance[contribution1])

			for j in range(0, len(contributions)):
				if j != i:
					contribution2 = contributions[j]

					if block_importance[contribution1][contribution2] > 0:
						p2 = contribution2.get_participant().get_id()

						current_value = conversation.get_score(p1, p2)
						current_value += importance[contribution1] * block_importance[contribution1][contribution2]

						conversation.set_score(p1, p2, current_value)


	@staticmethod
	def evaluate_involvement(conversation: Conversation):
		cna_graph = conversation.container.graph
		importance = cna_graph.importance
		participants = conversation.get_participants()

		if (len(participants) == 0):
			return

		for contribution in conversation.get_contributions():
			p = contribution.get_participant()

			current_value = p.get_index(CsclIndices.SCORE)
			p.set_index(CsclIndices.SCORE, current_value + importance[contribution])

			# TODO add social KB

			current_value = p.get_index(CsclIndices.NO_CONTRIBUTION)
			p.set_index(CsclIndices.NO_CONTRIBUTION, current_value + 1)

	
	@staticmethod
	def evaluate_used_concepts(conversation: Conversation):
		participants = conversation.get_participants()
		
		for p in participants:
			for contribution in conversation.get_participant_contributions(p.get_id()):
				for word in contribution.get_words():
					if word.pos.value[0] == 'N':
						current_value = p.get_index(CsclIndices.NO_NOUNS)
						p.set_index(CsclIndices.NO_NOUNS, current_value + 1)
						print(p.get_index(CsclIndices.NO_NOUNS))
					if word.pos.value[0] == 'V':
						current_value = p.get_index(CsclIndices.NO_VERBS)
						p.set_index(CsclIndices.NO_VERBS, current_value + 1)


	@staticmethod
	def perform_sna(conversation: Conversation, needs_anonymization: bool):
		participants = conversation.get_participants()

		for i, p1 in enumerate(participants):
			for j, p2 in enumerate(participants):
				if i != j:
					current_value = p1.get_index(CsclIndices.OUTDEGREE)
					p1.set_index(CsclIndices.OUTDEGREE, current_value +
												conversation.get_score(p1.get_id(), p2.get_id()))

					current_value = p2.get_index(CsclIndices.INDEGREE)
					p2.set_index(CsclIndices.INDEGREE, current_value +
												conversation.get_score(p1.get_id(), p2.get_id()))
				else:
					current_value = p1.get_index(CsclIndices.OUTDEGREE)
					p1.set_index(CsclIndices.OUTDEGREE, current_value +
												conversation.get_score(p1.get_id(), p1.get_id()))


