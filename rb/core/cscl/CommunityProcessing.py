
from datetime import date

class CommunityProcessing:

	@staticmethod
	def determine_participant_contributions(community):
		for c in community.get_conversations():
			for p in c.get_participants():
				if p.get_contributions().get_blocks() != None and len(p.get_contributions().get_blocks()) != 0:
					index = community.get_participants().index(p)
					participant_to_update = None

					if index >= 0:
						participant_to_update = community.get_participants()[index]
					else:
						participant_to_update = Participant(p.get_name(), c)
						community.get_participants().append(participant_to_update)

					for block in p.get_contributions().get_blocks():
						contribution = Contribution(block)

						if contribution != None and contribution.get_time() != None and contribution.is_eligible(community.get_start_date(), community.get_end_date()):
							if community.get_first_contribution_date() != None:
								community.set_first_contribution_date(contribution.get_time())

							if contribution.get_time() < community.get_first_contribution_date():
								community.set_first_contribution_date(contribution.get_time())

								# gregorian calendar 2010
								date_base = date(2010, 1, 1)

							if community.get_last_contribution_date() is None:
								community.set_last_contribution_date(contribution.get_time())

							if contribution.get_time() > community.get_last_contribution_date():
								community.set_last_contribution_date(contribution.get_time())

							Block.add_block(participant_to_update.get_contributions(), block)
							Block.add_block(community.get_eligible_contributions(), block)
							if block.is_significant():
								Block.add_block(participant_to_update.get_significant_contributions(), block)

							indices = participant_to_update.get_indices()
							indices[CSCLIndices.NO_CONTRIBUTION] += 1

							for key, value in contribution.get_word_occurences().items():
								if key.get_POS() != None:
									if key.get_POS().startswith("N"):
										indices[CSCLIndices.NO_NOUNS] += value

									if key.get_POS().startswith("V"):
										indices[CSCLIndices.NO_VERBS] += value

		if community.get_start_date() is None:
			community.set_start_date(community.get_first_contribution_date())
		if community.get_end_date() is None:
			community.set_end_date(community.get_last_contribution_date())

	@staticmethod
	def determine_participation(community):
		size = len(community.get_participants())
		community.set_participant_contributions(size, size)

		for c in community.get_conversations():
			blocks = c.get_blocks()

			for i in range(len(blocks)):
				contribution = Contribution(blocks[i])

				if contribution != None and contribution.get_time() != None and contribution.is_eligible(community.get_start_date(), community.get_end_date()):
					p1 = contribution.get_participant()
					index1 = community.get_participants().index(p1)

					if index1 >= 0:
						community.get_participant_contributions()[index1][index1] += contribution.get_score()
						participant_to_update = community.get_participants()[index1]

						indices = participant_to_update.get_indices()
						indices[CSCLIndices.SCORE] += contribution.get_score()
						indices[CSCLIndices.SOCIAL_KB] += contribution.get_social_KB()

						for j in range(i):
							if c.get_prunned_block_distances()[i][j] != None:
								p2 = (Contribution(blocks[j])).get_participant()
								index2 = community.get_participants().index(p2)

								if index2 >= 0:
									added_KB = blocks[i].get_score() * c.get_prunned_block_distances()[i][j].get_cohesion()
									community.get_participant_contributions()[index1][index2] += added_KB

	@staticmethod
	def compute_SNA_metrics(community):
		ParticipantEvaluation.perform_SNA_participants(community.get_participants(), community.get_participant_contributions(), True)

		for c in community.get_conversations():
			p = None
			blocks = c.get_blocks()

			for i in range(len(blocks)):
				if blocks[i] != None:
					if p is None:
						p = (Contribution(blocks[i])).get_participant()
						participant_to_update = community.get_participants()[community.get_participants().index(p)]
						indices = participant_to_update.get_indices()

						indices[CSCLIndices.NO_NEW_THREADS] += 1
						indices[CSCLIndices.NEW_THREADS_OVERALL_SCORE] += c.get_score()
						indices[CSCLIndices.NEW_THREADS_CUMULATIVE_SOCIAL_KB] +=
							VectorAlgebra.sum_elements(c.get_social_KB_evolution())
						indices[CSCLIndices.AVERAGE_LENGTH_NEW_THREADS] +=
							len(blocks[i].get_text())

						break

		filtered_participants = list(filter(lambda p: (p.get_indices()[CSCLIndices.NO_NEW_THREADS] != 0),
									community.get_participants()))

		for p in filtered_participants:
			indices = p.get_indices()
			indices[CSCLIndices.AVERAGE_LENGTH_NEW_THREADS] /= indices[CSCLIndices.NO_NEW_THREADS]


	
