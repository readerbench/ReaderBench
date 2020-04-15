from typing import List, Dict
import csv

from rb.core.lang import Lang
from rb.core.text_element import TextElement
from rb.core.text_element_type import TextElementType
from rb.core.cscl.contribution import Contribution
from rb.core.cscl.conversation import Conversation
from rb.core.cscl.community import Community
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


def main():
	conversation_thread1 = CsvParser.get_json_from_csv('./thread1.csv')
	conversation_thread2 = CsvParser.get_json_from_csv('./thread2.csv')

	#print(conversation_thread1)

	#conversation = Conversation(lang=Lang.EN, container=None,conversation_thread=conversation_thread1)
		
	#print(conversation.get_words())
	#contr = conversation.get_contributions()

	#for c in contr:
		#print(c.get_timestamp())


	community = Community(lang=Lang.EN,container=None,community=[conversation_thread1, conversation_thread2])

	print(community.get_sentences())
	print(community.get_conversations())

	l = community.get_participants()
	print(l)

	for p in l:
		print(community.get_participant_contributions(p.get_id()))

	print(community.get_first_contribution_date())
	print(community.get_last_contribution_date())

	c = community.get_conversations()[0]

	contr = c.get_contributions()

	for ct in contr:
		print(ct.get_participant().get_id())

	community.graph.compute_block_importance()

	conv = community.get_conversations()[0]

	ParticipantEvaluation.evaluate_interaction(conv)

	print(conv.get_score('name1', 'name2'))
	print(conv.get_score('name1', 'name3'))
	print(conv.get_score('name1', 'name4'))
	#print(conv.get_score('name1', 'name5'))
	#print(conv.get_score('name1', 'name6'))

	ParticipantEvaluation.evaluate_involvement(conv)
	ParticipantEvaluation.evaluate_used_concepts(conv)
	ParticipantEvaluation.perform_sna(conv, False)

if __name__ == '__main__':
	main()
