import csv
import json
from datetime import datetime
from typing import Dict, List

import xmltodict
from dateutil import parser
from rb.cna.cna_graph import CnaGraph
from rb.core.cscl.community import Community
from rb.core.cscl.contribution import Contribution
from rb.core.cscl.conversation import Conversation
from rb.core.cscl.cna_indices_enum import CNAIndices
from rb.core.cscl.participant import Participant
from rb.core.lang import Lang
from rb.core.text_element import TextElement
from rb.core.text_element_type import TextElementType
from rb.processings.cscl.community_processing import (
    compute_sna_metrics, determine_participant_contributions,
    determine_participation)
from rb.processings.cscl.participant_evaluation import (evaluate_interaction,
                                                        evaluate_involvement,
                                                        evaluate_textual_complexity,
                                                        perform_sna)
from rb.similarity.vector_model import VectorModelType
from rb.similarity.vector_model_factory import create_vector_model
from rb.utils.rblogger import Logger

logger = Logger.get_logger()

CONTRIBUTIONS_KEY = 'contributions'
ID_KEY = 'id'
PARENT_ID_KEY = 'parent_id'
TIMESTAMP_KEY = 'timestamp'
TEXT_KEY = 'text'
USER_KEY = 'user'

JSONS_PATH = './jsons/'

def get_json_from_json_file(filename: str) -> Dict:
    conversation_thread = dict()
    contribution_list = []

    with open(filename, "rt", encoding='utf-8') as json_file:
        contribution_list_json = json.load(json_file)
        for cnt in contribution_list_json:
            contribution = {
                ID_KEY: cnt['genid'],
                PARENT_ID_KEY: cnt["ref"],
                TIMESTAMP_KEY: int(float(cnt['time'])),
                USER_KEY: cnt["nickname"],
                TEXT_KEY: cnt["text"],
            }
            contribution_list.append(contribution)

    conversation_thread[CONTRIBUTIONS_KEY] = contribution_list

    return conversation_thread

def get_json_from_csv(filename: str) -> Dict:
    conversation_thread = dict()
    contribution_list = []

    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        first = True			

        for row in csv_reader:
            if first:
                first = False
            else:
                contribution = dict()

                contribution[ID_KEY] = row[0]
                contribution[PARENT_ID_KEY] = row[1]
                contribution[USER_KEY] = row[2]
                contribution[TEXT_KEY] = row[3]
                contribution[TIMESTAMP_KEY] = row[4]

                contribution_list.append(contribution)

    conversation_thread[CONTRIBUTIONS_KEY] = contribution_list

    return conversation_thread

def parse_large_csv(filename: str) -> Dict:
    conversation_thread = dict()
    contribution_list = []

    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        first = True			

        for row in csv_reader:
            if first:
                first = False
            else:
                contribution = dict()

                contribution[ID_KEY] = row[0]
                contribution[PARENT_ID_KEY] = row[1]
                if row[1] == '':
                    contribution[PARENT_ID_KEY] = '-1'

                contribution[USER_KEY] = row[2]
                contribution[TEXT_KEY] = row[6]

                contribution[TIMESTAMP_KEY] = datetime.timestamp(parser.parse(row[3], ignoretz=True, fuzzy=True))

                contribution_list.append(contribution)

    conversation_thread[CONTRIBUTIONS_KEY] = contribution_list

    return conversation_thread

def load_from_xml(lang: Lang, filename: str) -> Dict:
	with open(filename, "rt") as f:
		my_dict=xmltodict.parse(f.read())
		contributions = [
			{
				ID_KEY: int(utterance["@genid"]) - 1,
				PARENT_ID_KEY: int(utterance["@ref"]) - 1,
				TIMESTAMP_KEY: datetime.timestamp(datetime.strptime(utterance["@time"],'%H.%M.%S').replace(year=2020)),
				USER_KEY: turn["@nickname"],
				TEXT_KEY: utterance["#text"],
			}
			for turn in my_dict["corpus"]["Dialog"]["Body"]["Turn"]
			for utterance in (turn["Utterance"] if isinstance(turn["Utterance"], List) else [turn["Utterance"]])
		]

		return {CONTRIBUTIONS_KEY: contributions}


def export_individual_statistics(participants: List[Participant], filename: str):
	first = True

	cscl_keys = [CNAIndices.CONTRIBUTIONS_SCORE, CNAIndices.SOCIAL_KB, CNAIndices.NO_CONTRIBUTION, CNAIndices.OUTDEGREE, CNAIndices.INDEGREE,
				CNAIndices.NO_NEW_THREADS, CNAIndices.NEW_THREADS_OVERALL_SCORE, CNAIndices.NEW_THREADS_CUMULATIVE_SOCIAL_KB,
				CNAIndices.AVERAGE_LENGTH_NEW_THREADS]

	evaluate_interaction(conv)
	evaluate_involvement(conv)
	evaluate_textual_complexity(conv)
	perform_sna(conv, False)
	with open(filename, 'w') as f:
		for p in participants:
			indices = p.indices
			keys = ['participant'] + [str(k.value) for k in CNAIndices]
			values = [p.get_id()] + [str(p.get_index(k)) for k in cscl_keys]

			if first:
				f.write(','.join(keys) + '\n')
				first = False

			f.write(','.join(values) + '\n')
		f.flush()
		f.close()

def export_textual_complexity(participants: List[Participant], filename: str):
	first = True

	with open(filename, 'w') as f:
		for p in participants:
			indices = p.textual_complexity_indices
			keys = ['participant'] + [str(k) for k in indices.keys()]
			values = [p.get_id()] + [str(indices[k]) for k in indices.keys()]

			if first:
				f.write(','.join(keys) + '\n')
				first = False

			f.write(','.join(values) + '\n')
		f.flush()
		f.close()

def test_community_processing():
	print('Testing Community Processing')

	conv_thread = []
	for i in range(1, 20):
		conversation = get_json_from_json_file("./jsons/conversation_" + str(i) + ".json")
		conv_thread.append(conversation)

	community = Community(lang=Lang.EN, container=None, community=conv_thread)
	en_coca_word2vec = create_vector_model(Lang.EN, VectorModelType.from_str("word2vec"), "coca")
	community.graph = CnaGraph(docs=[community], models=[en_coca_word2vec])

	participant_list = community.get_participants()
	names = list(map(lambda p: p.get_id(), participant_list))

	print('Participants are:')
	print(names)

	print('Begin computing indices')

	determine_participant_contributions(community)
	determine_participation(community)
	compute_sna_metrics(community)
	# determine_textual_complexity(community)

	print('Finished computing indices')

	for p in participant_list:
		print('Printing for participant ' + p.get_id())

		print(p.get_index(CNAIndices.CONTRIBUTIONS_SCORE))
		print(p.get_index(CNAIndices.NO_CONTRIBUTION))
		print(p.get_index(CNAIndices.SOCIAL_KB))
		print(p.get_index(CNAIndices.INDEGREE))
		print(p.get_index(CNAIndices.OUTDEGREE))
		print(p.get_index(CNAIndices.NO_NEW_THREADS))
		print(p.get_index(CNAIndices.NEW_THREADS_OVERALL_SCORE))
		print(p.get_index(CNAIndices.NEW_THREADS_CUMULATIVE_SOCIAL_KB))
		print(p.get_index(CNAIndices.AVERAGE_LENGTH_NEW_THREADS))

		# print('Printing textual complexity for participant ' + p.get_id())
		# indices = p.textual_complexity_indices
		#
		# for key, value in indices.items():
		# 	print('Index ' + key + ' is ' + str(value))

		print('---------------------')

	for n1 in names:
		for n2 in names:
			print('Score for ' + n1 + ' ' + n2 + ' is:')
			print(community.get_score(n1, n2))

	export_individual_statistics(community.get_participants(), './individualStats.csv')
	# export_textual_complexity(community.get_participants(), './textualComplexity.csv')

def test_participant_evaluation():
	print('Testing English CSV')

	conv_thread = []
	for i in range(1, count_jsons + 1):
		conversation = get_json_from_json_file(JSONS_PATH + "conversation_" + str(i) + ".json")
		conv_thread.append(conversation)

	community = Community(lang=Lang.EN, container=None, community=conv_thread)
	en_coca_word2vec = create_vector_model(Lang.EN, VectorModelType.from_str("word2vec"), "coca")
	community.graph = CnaGraph(docs=[community], models=[en_coca_word2vec])

	conv = community.get_conversations()[0]

	# participant_list = conv.get_participants()
	participant_list = community.get_participants()
	names = list(map(lambda p: p.get_id(), participant_list))

	print('Participants are:')
	print(names)

	conv_thread = parse_large_csv('./thread.csv')
	print('Begin computing indices')

	evaluate_interaction(conv)
	evaluate_involvement(conv)
	# evaluate_textual_complexity(conv)
	perform_sna(conv, False)

	print('Finished computing indices')

	for p in participant_list:
		print('Printing for participant ' + p.get_id())

		print(p.get_index(CNAIndices.CONTRIBUTIONS_SCORE))
		print(p.get_index(CNAIndices.NO_CONTRIBUTION))
		print(p.get_index(CNAIndices.SOCIAL_KB))
		print(p.get_index(CNAIndices.INDEGREE))
		print(p.get_index(CNAIndices.OUTDEGREE))

		# print('Printing textual complexity for participant ' + p.get_id())
		# indices = p.textual_complexity_indices
		#
		# for key, value in indices.items():
		# 	print('Index ' + key + ' is ' + str(value))

		print('---------------------')

	for n1 in names:
		for n2 in names:
			print('Score for ' + n1 + ' ' + n2 + ' is:')
			print(conv.get_score(n1, n2))

	export_individual_statistics(conv.get_participants(), './individualStats.csv')

def main():
	# Test community processing - English discussion csv
	test_community_processing()

	# Test participant evaluation - English discussion csv
	#test_participant_evaluation()

	# Test for French discussion XML

	'''

	print('Testing French XML')

	conv_thread = CsvParser.load_from_xml(Lang.FR, './conpa-MEEF-anonyme.xml')
	
	community = Community(lang=Lang.FR, container=None, community=[conv_thread])
	fr_coca_word2vec = create_vector_model(Lang.FR, VectorModelType.from_str("word2vec"), "coca")
	community.graph = CnaGraph(docs=[community], models=[fr_coca_word2vec])

	conv = community.get_conversations()[0]

	compute_indices(conv)

	'''

if __name__ == '__main__':
	main()
