from typing import List, Dict
import csv

from rb.core.lang import Lang
from rb.core.text_element import TextElement
from rb.core.text_element_type import TextElementType
from rb.core.cscl.contribution import Contribution
from rb.core.cscl.conversation import Conversation

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
	conversation_thread = CsvParser.get_json_from_csv('./thread.csv')

	print(conversation_thread)

	#conversation = Conversation(lang=Lang.EN, container=None,
	#							conversation_thread=conversation_thread)

if __name__ == '__main__':
	main()
