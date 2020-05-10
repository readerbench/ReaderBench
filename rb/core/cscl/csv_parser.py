from typing import List, Dict
import csv
from dateutil import parser
from datetime import datetime
import xmltodict

from rb.core.lang import Lang
from rb.core.text_element import TextElement
from rb.core.text_element_type import TextElementType
from rb.core.cscl.contribution import Contribution
from rb.core.cscl.conversation import Conversation
from rb.core.cscl.community import Community
from rb.core.cscl.cscl_indices import CsclIndices
from rb.cna.cna_graph import CnaGraph
from rb.similarity.vector_model_factory import create_vector_model
from rb.similarity.vector_model import VectorModelType
from rb.processings.cscl.participant_evaluation import ParticipantEvaluation

from rb.utils.rblogger import Logger

logger = Logger.get_logger()

CONTRIBUTIONS_KEY = 'contributions'
ID_KEY = 'id'
PARENT_ID_KEY = 'parent_id'
TIMESTAMP_KEY = 'timestamp'
TEXT_KEY = 'text'
USER_KEY = 'user'

class CsvParser:

	@staticmethod
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

	@staticmethod
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

	@staticmethod
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

def compute_indices(conv: Conversation):
	participant_list = conv.get_participants()
	names = list(map(lambda p: p.get_id(), participant_list))

	print('Participants are:')
	print(names)

	print('Begin computing indices')

	ParticipantEvaluation.evaluate_interaction(conv)
	ParticipantEvaluation.evaluate_involvement(conv)
	ParticipantEvaluation.evaluate_used_concepts(conv)
	ParticipantEvaluation.perform_sna(conv, False)

	print('Finished computing indices')

	for p in participant_list:
		print('Printing for participant ' + p.get_id())

		print(p.get_index(CsclIndices.SCORE))
		print(p.get_index(CsclIndices.NO_NOUNS))
		print(p.get_index(CsclIndices.NO_VERBS))
		print(p.get_index(CsclIndices.NO_CONTRIBUTION))
		print(p.get_index(CsclIndices.SOCIAL_KB))
		print(p.get_index(CsclIndices.INDEGREE))
		print(p.get_index(CsclIndices.OUTDEGREE))

		print('---------------------')

	for n1 in names:
		for n2 in names:
			print('Score for ' + n1 + ' ' + n2 + ' is:')
			print(conv.get_score(n1, n2))

def main():
	# Test for English discussion CSV

	print('Testing English CSV')

	conv_thread = CsvParser.parse_large_csv('./thread.csv')

	community = Community(lang=Lang.EN, container=None, community=[conv_thread])
	en_coca_word2vec = create_vector_model(Lang.EN, VectorModelType.from_str("word2vec"), "coca")
	community.graph = CnaGraph(docs=[community], models=[en_coca_word2vec])

	conv = community.get_conversations()[0]

	compute_indices(conv)

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
