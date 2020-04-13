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
		block_cohesion = cna_graph.block_cohesion

		participants = conversation.get_participants()
		contributions = conversation.get_contributions()
		conversation.init_scores()

		if (len(participants) == 0):
			return

		for i, contribution1 in enumerate(contributions):
			p1 = contribution1.get_participant_id()

			conversation.update_score(p1, p1, importance[contribution1])

			for j in range(0, len(contributions)):
				if j != i:
					contribution2 = contributions[j]

					if block_cohesion[contribution1][contribution2] > 0:
						p2 = contribution2.get_participant_id()
						conversation.update_score(p1, p2,
							importance[contribution1] * block_cohesion[contribution1][contribution2])

	@staticmethod
	def evalute_involvement(conversation: Conversation):
		cna_graph = conversation.container.graph
		importance = cna_graph.importance
		participants = conversation.get_participants()

		if (len(participants) == 0):
			return:

		conversation.init_indices()

		for contribution in conversation.get_contributions():
			p = contribution.get_participant_id()

			current_value = conversation.get_index(p, CsclIndices.SCORE)
			conversation.update_index(p, CsclIndices.SCORE, current_value + importance[contribution])

			# TODO add social kb

			current_value = conversation.get_index(p, CsclIndices.NO_CONTRIBUTION)
			conversation.update_index(p, CsclIndices.NO_CONTRIBUTION, current_value + 1)


	@staticmethod
	def evaluate_used_concept(conversation: Conversation):
		participants = conversation.get_participants()

		for p in participants:
			for contribution in conversation.get_participant_contributions(p):
				for word in contribution.get_words()
					if word.pos[0] == 'N':
						current_value = conversation.get_index(p, CsclIndices.NO_NOUNS)
						conversation.update_index(p, CsclIndices.NO_NOUNS, current_value + 1)
					if word.pos[0] == 'V':
						current_value = conversation.get_index(p, CsclIndices.NO_VERBS)
						conversation.update_index(p, CsclIndices.NO_VERBS, current_value + 1)

	@staticmethod
	def perform_sna(conversation: Conversation, needs_anonymization: bool):
		participants = conversation.get_participants()

		for i, p1 in enumerate(participants):
			for j, p2 in enumerate(participants):
				if i != j:
					current_value = conversation.get_index(p1, CsclIndices.OUTDEGREE)
					conversation.update_index(p1, CsclIndices.OUTDEGREE, current_value +
												conversation.get_score(p1, p2))

					current_value = conversation.get_index(p2, CsclIndices.INDEGREE)
					conversation.update_index(p2, CsclIndices.INDEGREE, current_value +
												conversation.get_score(p1, p2))
				else:
					current_value = conversation.get_index(p1, CsclIndices.OUTDEGREE)
					conversation.update_index(p1, CsclIndices.OUTDEGREE, current_value +
												conversation.get_score(p1, p1))




