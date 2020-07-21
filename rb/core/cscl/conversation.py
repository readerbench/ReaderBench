import json
from typing import Dict, List

import xmltodict
from copy import deepcopy

from rb.core.block import Block
from rb.core.cscl.contribution import Contribution
from rb.core.document import Document
from rb.core.lang import Lang
from rb.core.sentence import Sentence
from rb.core.text_element import TextElement
from rb.core.text_element_type import TextElementType
from rb.core.cscl.participant import Participant
from rb.utils.rblogger import Logger

logger = Logger.get_logger()

CONTRIBUTIONS_KEY = 'contributions'
ID_KEY = 'id'
PARENT_ID_KEY = 'parent_id'
TIMESTAMP_KEY = 'timestamp'
TEXT_KEY = 'text'
USER_KEY = 'user'

TIMEFRAME = 30
DISTANCE = 20

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
				 apply_heuristics: bool = True
                 ):

		TextElement.__init__(self, lang=lang, text="",
                             depth=depth, container=container)

		self.participant_map = dict()
		self.participants = []
		self.participant_contributions = dict()

		self.parse_contributions(conversation_thread, apply_heuristics=apply_heuristics)

		self.scores = dict()
		self.init_scores()


	def time_heuristic(self, contributions: List[Dict]) -> List[Dict]:
		last_contribution = dict()
		group = dict()
		processed_contributions = []

		for i, contribution in enumerate(contributions):
			user = contribution[USER_KEY]
			timestamp = contribution[TIMESTAMP_KEY]

			if user in last_contribution and ((timestamp - last_contribution[user][TIMESTAMP_KEY]) <= TIMEFRAME):
				last = last_contribution[user]
				last[TEXT_KEY] += (' ' + contribution[TEXT_KEY])
				group[i] = last

			else:
				contribution[ID_KEY] = len(processed_contributions)
				group[i] = contribution

				processed_contributions.append(contribution)
				last_contribution[user] = contribution
			
				parent_id = int(contribution[PARENT_ID_KEY])
				if parent_id > 0:
					parent = group[parent_id]

					contribution[PARENT_ID_KEY] = parent[ID_KEY]

		return processed_contributions

	def distance_heuristic(self, contributions: List[Dict]) -> List[Dict]:
		last_contribution = dict()
		last_contribution_index = dict()
		group = dict()
		processed_contributions = []

		for i, contribution in enumerate(contributions):
			user = contribution[USER_KEY]

			if user in last_contribution_index and ((i - last_contribution_index[user]) <= DISTANCE):
				last = last_contribution[user]
				last[TEXT_KEY] += (' ' + contribution[TEXT_KEY])
				group[i] = last

			else:
				last_contribution_index[user] = int(contribution[ID_KEY])
				contribution[ID_KEY] = len(processed_contributions)
				group[i] = contribution

				processed_contributions.append(contribution)
				last_contribution[user] = contribution
			
				parent_id = int(contribution[PARENT_ID_KEY])
				if parent_id > 0:
					parent = group[parent_id]

					contribution[PARENT_ID_KEY] = parent[ID_KEY]

		return processed_contributions

	def parse_contributions(self, conversation_thread: Dict, apply_heuristics: bool = True):
		contribution_map = dict()
		users = set()

		full_text = ''

		# apply both heuristics before processing
		contributions = conversation_thread[CONTRIBUTIONS_KEY]

		if apply_heuristics:
			contributions = self.time_heuristic(contributions)
			contributions = self.distance_heuristic(contributions)

		self.participants = []

		for contribution in contributions:
			index = int(contribution[ID_KEY])
			participant_id = contribution[USER_KEY]
			users.add(participant_id)

			# parent index will be -1 in JSON for the first post
			parent_index = int(contribution[PARENT_ID_KEY])

			timestamp = int(contribution[TIMESTAMP_KEY])
			text = contribution[TEXT_KEY].strip()
			if text[-1] not in {'.', '!', '?'}:
				text += "."
			
			parent_contribution = None

			if parent_index > 0:
				parent_contribution = contribution_map[parent_index]

			full_text += text + "\n"

			participant = None

			if not (participant_id in self.participant_map):
				participant = Participant(participant_id=participant_id)

				self.participant_map[participant_id] = participant
				self.participants.append(participant)
			else:
				participant = self.participant_map[participant_id]

			if not (self.container is None):
				if not (participant_id in self.container.participant_map):
					global_participant = Participant(participant_id=participant_id)
					self.container.participant_map[participant_id] = global_participant		

			current_contribution = Contribution(self.lang, text, container=self,
												participant=participant,
												parent_contribution=parent_contribution,
												contribution_raw=contribution,
												timestamp=timestamp)

			self.components.append(current_contribution)
			contribution_map[index] = current_contribution

			if not (participant_id in self.participant_contributions):
				self.participant_contributions[participant_id] = []

			self.participant_contributions[participant_id].append(current_contribution)

		self.text = full_text

		sentences = self.parse_full_text(full_text)
		i = 0
		for contribution in self.components:
			left = len(contribution.text)
			if "\n" in sentences[i].text[:-1]:
				print("aici")
			while i < len(sentences) and len(sentences[i].text) <= left:
				contribution.add_sentence(sentences[i])
				left -= len(sentences[i].text)
				i += 1


	def parse_full_text(self, full_text: str) -> List[Sentence]:
		parsed_document = Block(self.lang, full_text)
		return parsed_document.get_sentences()

	def get_participants(self) -> List[Participant]:
		return self.participants

	def get_participant_contributions(self, participant_id: str) -> List[Contribution]:
		return self.participant_contributions[participant_id]

	def get_contributions(self) -> List[Contribution]:
		return self.components

	def init_scores(self):
		for a in self.participants:
			self.scores[a.get_id()] = dict()

			for b in self.participants:
				self.scores[a.get_id()][b.get_id()] = 0

	def get_score(self, a: str, b: str) -> float:
		return self.scores[a][b]

	def update_score(self, a: str, b: str, value: float):
		self.scores[a][b] += value

	def __str__(self):
		return NotImplemented

	@staticmethod
	def create_participant_conversation(lang: Lang, p: Participant) -> "Conversation":
		conversation_thread = dict()
		contribution_list = []

		index = 0
		old_index = dict()

		for c in p.eligible_contributions:
			contribution = deepcopy(c.get_raw_contribution())

			old_index[contribution[ID_KEY]] = index
			contribution[ID_KEY] = index

			if contribution[PARENT_ID_KEY] in old_index:
				contribution[PARENT_ID_KEY] = old_index[contribution[PARENT_ID_KEY]]
			else:
				contribution[PARENT_ID_KEY] = -1

			contribution_list.append(contribution)
			index += 1

		conversation_thread[CONTRIBUTIONS_KEY] = contribution_list

		conversation = Conversation(lang=lang, container=None,
									conversation_thread=conversation_thread,
									apply_heuristics=False)

		p.set_own_conversation(conversation)

	
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