from typing import Dict, List

from rb.core.lang import Lang
from rb.core.text_element import TextElement
from rb.core.text_element_type import TextElementType
from rb.utils.rblogger import Logger
import csv

class Participant:

    def __init__(self, participant_id: str):
        self.participant_id = participant_id
        self.indices = dict()
        self.longitudinal_indices = dict()
        self.textual_complexity_indices = dict()

        self.eligible_contributions = []
        self.significant_contributions = []

        self.own_conversation = None

    def get_id(self) -> str:
        return self.participant_id

    def get_index(self, index: str) -> float:
        if index in self.indices:
            return self.indices[index]

        return 0
        
    def get_textual_index(self, index: str) -> float:
        if index in self.textual_complexity_indices:
            return self.textual_complexity_indices[index]
        return 0

    def __eq__(self, other):
        return self.participant_id == other.participant_id
    
    def __hash__(self):
        return hash(self.participant_id)

    def set_index(self, index: str, value: float):
        self.indices[index] = value

    def set_longitudinal_index(self, index: str, value: float):
        self.longitudinal_indices[index] = value

    def set_textual_index(self, index: str, value: float):
        self.textual_complexity_indices[index] = value

    def set_own_conversation(self, conversation):
        self.own_conversation = conversation

    def add_eligible_contribution(self, contribution):
        self.eligible_contributions.append(contribution)

    def set_eligible_contributions(self, contributions):
        self.eligible_contributions = contributions

    def add_significant_contribution(self, contribution):
        self.significant_contributions.append(contribution)

    def export_individual_statistics(self, filename: str):
        with open('mycsvfile.csv','wb') as f:
            w = csv.writer(f)
            w.writerow(somedict.keys())
            w.writerow(somedict.values())
