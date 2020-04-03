from typing import List, Dict

from rb.core.lang import Lang
from rb.core.text_element import TextElement
from rb.core.text_element_type import TextElementType
from rb.core.cscl.contribution import Contribution
from rb.core.cscl.conversation import Conversation

from rb.utils.rblogger import Logger

logger = Logger.get_logger()

# muchii intre replici parinte-copil?
# muchii intre replici ale aceluiasi user ?

class Community(TextElement):

	'''
		community -> list of JSON objects (crawled from Reddit), with
					 the same structure as described in Conversation class
	'''
	def __init__(self, lang: Lang, 
                 community: List[Dict],
				 container: TextElement = None,
                 depth: int = TextElementType.COMM.value,
                 start_date: int = None,
                 end_date: int = None):

		TextElement.__init__(self, lang=lang, text=None,
                             depth=depth, container=container)

		self.components = []
		self.start_date = start_date
		self.end_date = end_date

		for conversation in community:
			current_contribution = Contribution(lang, text, container=self,
						 						conversation_thread=conversation_thread)

			self.components.append(current_contribution)

		self.participants = union_participants()
		self.participant_contributions = union_contributions()
		self.first_contribution_date, self.last_contribution_date = find_contribution_range()


	def union_participants(self) -> List[str]:
		all_participants = set()

		for conversation in self.components:
			for participant in conversation.get_participants():
				all_participants.add(participant)

		return list(all_participants)

	def union_contributions(self) -> Dict:
		contributions = dict()

		for conversation in self.components:
			for participant in conversation.get_participants():
				if not (participant in contributions):
					contributions[participant] = []

				contributions[participant] += conversation.get_participant_contributions(participant)

		return contributions

	def find_contribution_range(self) -> int, int:
		first_contribution = None
		last_contribution = None

		for conversation in self.components:
			for contribution in conversation.get_contributions():
				timestamp = contribution.get_timestamp()

				if first_contribution == None or timestamp < first_contribution:
					first_contribution = timestamp

				if last_contribution == None or timestamp > last_contribution:
					last_contribution = timestamp

		return first_contribution, last_contribution

	def get_conversations(self) -> List[Conversation]:
		return self.components

	def get_participants(self) -> List[str]:
		return self.participants

	def get_participant_contributions(self, participant_id: str) -> List[Contribution]:
		return self.participant_contributions[participant_id]

	def get_first_contribution_date(self) -> int:
		return self.first_contribution_date

	def get_last_contribution_date(self) -> int:
		return self.last_contribution_date