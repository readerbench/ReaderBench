from typing import Dict, List, Tuple

from rb.core.lang import Lang
from rb.core.text_element import TextElement
from rb.core.text_element_type import TextElementType
from rb.core.cscl.contribution import Contribution
from rb.core.cscl.conversation import Conversation
from rb.core.cscl.participant import Participant
from rb.similarity.vector_model_factory import create_vector_model
from rb.similarity.vector_model import VectorModelType

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

		self.participant_map = dict()

		for conversation_thread in community:
			current_conversation = Conversation(lang, container=self,
						 						conversation_thread=conversation_thread)

			self.components.append(current_conversation)

		self.participants = self.union_participants()
		self.participant_contributions = self.union_contributions()
		self.first_contribution_date, self.last_contribution_date = self.find_contribution_range()

		self.graph = None

		self.eligible_contributions = []
		self.timeframe_subcommunities = []
		self.scores = dict()
		self.init_scores()

		self.community_raw = community

	def union_participants(self) -> List[Participant]:
		return [self.participant_map[participant_id] for participant_id in self.participant_map]

	def union_contributions(self) -> Dict:
		contributions = dict()

		for conversation in self.components:
			for participant in conversation.get_participants():
				if not (participant in contributions):
					contributions[participant.get_id()] = []

				contributions[participant.get_id()] += conversation.get_participant_contributions(participant.get_id())

		return contributions

	def get_community_raw(self) -> List[Dict]:
		return self.community_raw

	def find_contribution_range(self) -> Tuple[int, int]:
		first_contribution = None
		last_contribution = None

		for conversation in self.components:
			for contribution in conversation.get_contributions():
				timestamp = contribution.get_timestamp()

				if self.is_eligible(timestamp):
					if first_contribution == None or timestamp < first_contribution:
						first_contribution = timestamp

					if last_contribution == None or timestamp > last_contribution:
						last_contribution = timestamp

		if self.start_date is None:
			self.start_date = first_contribution
		if self.end_date is None:
			self.end_date = last_contribution

		return first_contribution, last_contribution

	def is_eligible(self, timestamp: int) -> bool:
		if self.start_date != None and self.end_date != None:
			return (timestamp >= self.start_date and timestamp <= self.end_date)
		
		return True

	def add_eligible_contribution(self, contribution: Contribution):
		self.eligible_contributions.append(contribution)

	def add_subcommunity(self, community):
		self.timeframe_subcommunities.append(community)

	def get_conversations(self) -> List[Conversation]:
		return self.components

	def get_participant(self, participant_id: str) -> Participant:
		return self.participant_map[participant_id]

	def get_participants(self) -> List[Participant]:
		return self.participants

	def get_participant_contributions(self, participant_id: str) -> List[Contribution]:
		return self.participant_contributions[participant_id]

	def get_first_contribution_date(self) -> int:
		return self.first_contribution_date

	def get_last_contribution_date(self) -> int:
		return self.last_contribution_date

	def init_scores(self):
		for a in self.participants:
			self.scores[a.get_id()] = dict()

			for b in self.participants:
				self.scores[a.get_id()][b.get_id()] = 0

	def get_score(self, a: str, b: str) -> float:
		return self.scores[a][b]

	def update_score(self, a: str, b: str, value: float):
		self.scores[a][b] += value
