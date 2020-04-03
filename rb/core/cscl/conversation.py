import json
from typing import Dict, List

import xmltodict
from rb.core.block import Block
from rb.core.cscl.contribution import Contribution
from rb.core.document import Document
from rb.core.lang import Lang
from rb.core.sentence import Sentence
from rb.core.text_element import TextElement
from rb.core.text_element_type import TextElementType
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
	def __init__(self, lang: Lang, conversation_thread: Dict,
				container: TextElement = None,
                 depth: int = TextElementType.DOC.value,
                 ):

		TextElement.__init__(self, lang=lang, text="",
                             depth=depth, container=container)

		self.participants = []
		self.participant_contributions = dict()

		self.parse_contributions(conversation_thread)


	def parse_contributions(self, conversation_thread: Dict):
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
			text = contribution[TEXT_KEY].strip()
			if text[-1] not in {'.', '!', '?'}:
				text += "."
			
			parent_contribution = None

			if parent_index > 0:
				parent_contribution = contribution_map[parent_index]

			full_text += text + "\n"
			current_contribution = Contribution(self.lang, text, container=self,
												participant_id=participant_id,
												parent_contribution=parent_contribution,
												timestamp=timestamp)

			self.components.append(current_contribution)
			contribution_map[index] = current_contribution

			if not (participant_id in self.participant_contributions):
				self.participant_contributions[participant_id] = []

			self.participant_contributions[participant_id].append(current_contribution)
	
		self.participants = list(users)
		self.text = full_text
		# way must be found to re-map parsed sentences to their original contribution
		sentences = self.parse_full_text(full_text)
		i = 0
		for contribution in self:
			left = len(contribution.text)
			if "\n" in sentences[i].text[:-1]:
				print("aici")
			while i < len(sentences) and len(sentences[i].text) <= left:
				contribution.add_sentence(sentences[i])
				left -= len(sentences[i].text)
				i += 1
		print("end")




	def parse_full_text(self, full_text: str) -> List[Sentence]:
		parsed_document = Block(self.lang, full_text)
		return parsed_document.get_sentences()

	def get_participants(self) -> List[str]:
		return self.participants

	def get_participant_contributions(self, participant_id: str) -> List[Contribution]:
		return self.participant_contributions[participant_id]

	def get_contributions(self) -> List[Contribution]:
		return self.components

	def __str__(self):
		return NotImplemented
	
	@staticmethod
	def load_from_xml(lang: Lang, filename: str) -> "Conversation":
		with open(filename, "rt") as f:
			my_dict=xmltodict.parse(f.read())
			contributions = [
				{
					ID_KEY: int(utterance["@genid"]),
					PARENT_ID_KEY: int(utterance["@ref"]),
					TIMESTAMP_KEY: utterance["@time"],
					USER_KEY: turn["@nickname"],
					TEXT_KEY: utterance["#text"],
				}
				for turn in my_dict["corpus"]["Dialog"]["Body"]["Turn"]
				for utterance in (turn["Utterance"] if isinstance(turn["Utterance"], List) else [turn["Utterance"]])
			]
			return Conversation(lang, {CONTRIBUTIONS_KEY: contributions})
