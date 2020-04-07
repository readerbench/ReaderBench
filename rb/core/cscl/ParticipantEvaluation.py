
from datetime import date

class ParticipantEvaluation:

	@staticmethod
	def evaluate_interaction(c):
		if (len(c.get_participants()) > 0):
			size = len(c.get_participants())

			c.set_participant_contributions(size, size)

			blocks = c.get_blocks()
			for i in range(len(blocks)):
				block = blocks[i]

				if block != None:
					p1 = Contribution(block).get_participant()
					index1 = c.get_participants().index(p1)
					c.get_participant_contributions()[index1][index1] += block.get_score()

					for j in range(i):
						if c.get_prunned_block_distances()[i][j] != None:
							p2 = Contribution(blocks[j]).get_participant()
							index2 = c.get_participants().index(p2)

							cohesion = c.get_prunned_block_distances()[i][j].get_cohesion()
							c.get_participant_contributions()[index1][index2] += block.get_score() * cohesion


	@staticmethod
	def evaluate_involvement(c):
		if (len(c.get_participants()) > 0):
			for block in c.get_blocks():
				if block != None:
					contribution = Contribution(block)
					indices = contribution.get_participant().get_indices()

					indices[CSCLIndices.SCORE] = indices[CSCLIndices.SCORE] + block.get_score()
					indices[CSCLIndices.SOCIAL_KB] = indices[CSCLIndices.SOCIAL_KB] + contribution.get_socialKB()
					indices[CSCLIndices.NO_CONTRIBUTION] = indices[CSCLIndices.NO_CONTRIBUTION] + 1

	@staticmethod
	def evaluate_used_concepts(c):
		for p in c.get_participants():
			word_occurences = p.get_contributions().get_word_occcurences()
			indices = p.get_indices()

			for key, value in word_occurences.items():
				if key.get_POS().startswith("N"):
					indices[CSCLIndices.NO_NOUNS] = indices[CSCLIndices.NO_NOUNS] + value

				if key.get_POS().startswith("V"):
					indices[CSCLIndices.NO_VERBS] = indices[CSCLIndices.NO_VERBS] + value

	@staticmethod
	def perform_SNA(c):
		perform_SNA_participants(c.get_participants(), c.get_participant_contributions(), True)

	@staticmethod
	def perform_SNA_participants(participants, participant_contributions, needs_anonymization):
		for index1 in range(len(participants)):
			for index2 in range(len(participants)):
				if index1 != index2:
					indices = participants[index1].get_indices()

					indices[CSCLIndices.OUTDEGREE] = indices[CSCLIndices.OUTDEGREE] +
						participant_contributions[index1][index2]

					indices = participants[index2].get_indices()

					indices[CSCLIndices.INDEGREE] = indices[CSCLIndices.INDEGREE] +
						participant_contributions[index1][index2]
				else:
					indices = participants[index1].get_indices()

					indices[CSCLIndices.OUTDEGREE] = indices[CSCLIndices.OUTDEGREE] +
						participant_contributions[index1][index2]

	@staticmethod
	def extract_rhythmic_index(c):
		# dictionary must be sorted
		rhythmic_ind_per_part = dict()

		if (len(c.get_participants()) > 0):
			for block in c.get_blocks():
				if block != None:
					contribution = Contribution(block)
					p = contribution.get_participant()

					if not (p in rhythmic_ind_per_part):
						rhythmic_ind_per_part[p] = []

					for s in contribution.get_sentences():
						unit = s.get_all_words()

						ind = RhythmTool.calc_rhythmic_index_SM(unit)
						if ind != RhythmTool.UNDEFINED:
							rhythmic_ind_per_part[p].append(ind)

		for key, value in rhythmic_ind_per_part.items():
			if len(value) == 0:
				continue

			max_index = max(value)
			key.get_indices()[CSCLIndices.RHYTHMIC_INDEX] = double(maxIndex)

			result = double(value.count(max_index)) / len(value)
			key.get_indices()[CSCLIndices.FREQ_MAX_RHYTHMIC_INDEX] = result

	@staticmethod
	def extract_rhythmic_coefficient(c):
		# dictionary must be sorted
		cnt_syllables = dict()
		deviations = dict()

		if (len(c.get_participants()) > 0):
			for block in c.get_blocks():
				if block != None:
					contribution = Contribution(block)
					p = contribution.get_participant()

					if not (p in cnt_syllables):
						# dictionary must be sorted
						cnt_syllables[p] = dict()

					for s in contribution.get_sentences():
						unit = s.get_all_words()
						representation = RhythmTool.get_numerical_representation(unit)

						if representation.is_empty():
							continue

						NT = len(representation) - 1 if (representation[0] == 0) else len(representation)
						NA = sum(representation)

						nr_sylls = cnt_syllables[p]
						for nr in representation:
							if nr == 0:
								continue

							nr_sylls[nr] = nr_sylls[nr] + 1 if (nr in nr_sylls) else 1

						devs = RhythmTool.calc_deviations(representation)
						deviations[p] = deviations[p] + 1 if (p in deviations) else devs

		for p in c.get_participants():
			nr_sylls = cnt_syllables[p]
			total_number = sum(nr_sylls.values())

			for key, value in nr_sylls.items():
				syll_freq = double(value) / total_number

			dominant_id = RhythmTool.get_dominant_index(list(nr_sylls.values()))
			key_of_max_val = list(nr_sylls.keys())[dominant_id]

			sylls_sum = nr_sylls[key_of_max_val]
			sylls_sum += nr_sylls[key_of_max_val - 1]  if ((key_of_max_val - 1) in nr_sylls) else 0
			sylls_sum += nr_sylls[key_of_max_val + 1]  if ((key_of_max_val + 1) in nr_sylls) else 0

			coeff = double(deviations[p] + total_number - sylls_sum) / total_number
			p.get_indices()[CSCLIndices.RHYTHMIC_COEFFICIENT, coeff]


	@staticmethod
	def compute_entropy_for_regularity_measure(c):
		chat_start_time = None
		chat_end_time = None
		chat_time = 0
		frame_time = 5 * 60

		# dictionary must be sorted
		timestamps = dict()

		for p in c.get_participants():
			dates = []

			for block in p.get_contributions().get_blocks():
				d = (Contribution(block)).timestamp()
				dates.append(d)

				if (chat_start_time is None and chat_end_time is None):
					chat_start_time = d
					chat_end_time = d
				else:
					if d < chat_start_time:
						chat_start_time = d
					if d > chat_end_time:
						chat_end_time = d

			timestamps[p] = dates

		chat_time = chat_end_time.timestamp() - chat_start_time.timestamp()

		# dictionary must be sorted
		no_interventions = dict()
		index = None
		diff = None

		size = int(ceil(double(chat_time) / frame_time))
		for key, value in timestamps.items():
			arr = [double(0)] * size

			for d in value:
				diff = d.timestamp() - chat_start_time.timestamp()
				index = int(floor((double(diff) / frame_time)))
				arr[index] = arr[index] + 1

			no_interventions[key] = arr

		for key, value in no_interventions.items():
			value = CSCLCriteria.get_value(CSCLCriteria.PEAK_CHAT_FRAME, list(map(double, value))
			key.get_indices()[CSCLCriteria.PERSONAL_REGULARITY_ENTROPY] = value

























