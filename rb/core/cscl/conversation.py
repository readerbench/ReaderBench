from typing import List, Dict

from rb.core.lang import Lang
from rb.core.text_element import TextElement
from rb.core.sentence import Sentence
from rb.core.document import Document
from rb.core.text_element_type import TextElementType
from rb.core.cscl.contribution import Contribution

from rb.utils.rblogger import Logger

logger = Logger.get_logger()

CONTRIBUTIONS_KEY = 'contributions'
ID_KEY = 'id'
PARENT_ID_KEY = 'parent_id'
TIMESTAMP_KEY = 'timestamp'
TEXT_KEY = 'text'
USER_KEY = 'user'

class Conversation(TextElement):

	'''
		conversation_thread -> JSON object (crawled from Reddit) parsed as Dict
		contains a list with one object per contribution
		each contribution contains: user (anonymous id)
									id
									parent_id
									timestamp
									text
	'''
	def __init__(self, lang: Lang, container: TextElement = None,
                 depth: int = TextElementType.DOC.value,
                 conversation_thread: Dict):

		TextElement.__init__(self, lang=lang, text=None,
                             depth=depth, container=container)

		self.participants = []
		self.participant_contributions = dict()

		parse_contributions(conversation_thread, lang)


	def parse_contributions(self, lang: Lang, conversation_thread: Dict):
		contribution_map = dict()
		users = set()

		full_text = ''

		for contribution in conversation_thread[CONTRIBUTIONS_KEY]:
			index = contribution[ID_KEY]
			participant_id = contribution[USER_KEY]
			users.add(participant_id)

			# parent index will be -1 in JSON for the first post
			parent_index = contribution[PARENT_ID_KEY]

			timestamp = contribution[TIMESTAMP_KEY]
			text = contribution[TEXT_KEY]

			parent_contribution = None

			if parent_index != -1:
				parent_contribution = contribution_map[parent_index]

			full_text += text

			current_contribution = Contribution(lang, text, container=self,
												participant_id=participant_id,
												parent_contribution=parent_contribution,
												timestamp=timestamp)

			self.components.append(current_contribution)
			contribution_map[index] = current_contribution

			if not (participant_id in self.participant_contributions):
				self.participant_contributions[participant_id] = []

			self.participant_contributions[participant_id].append(current_contribution)
	
		self.participants = list(users)

		# way must be found to re-map parsed sentences to their original contribution
		for sentence in parse_full_text(full_text):
			#contribution_map[participant_id].add_sentence(sentence)



	def parse_full_text(self, full_text: str, lang: Lang) -> List[Sentence]:
		parsed_document = Document(lang, text)

		return parsed_document.get_sentences()

	def get_participants(self) -> List[str]:
		return self.participants

	def get_participant_contributions(self, participant_id: str) -> List[Contribution]:
		return self.participant_contributions[participant_id]

    def get_contributions(self) -> List[Contribution]:
    	return self.components

    def __str__(self):
        return NotImplemented