from datetime import datetime
from typing import List, Dict

from rb.core.block import Block
from rb.core.lang import Lang
from rb.core.sentence import Sentence
from rb.core.text_element import TextElement
from rb.core.block import Block
from rb.core.text_element_type import TextElementType
from rb.core.cscl.participant import Participant
from rb.parser.spacy_parser import SpacyParser
from rb.utils.rblogger import Logger

logger = Logger.get_logger()

SIGNIFICANT_LIMIT = 5


class Contribution(TextElement):

    def __init__(self, lang: Lang, text: str, participant: Participant,
                 parent_contribution: "Contribution", contribution_raw: Dict, index: int,
                 timestamp: datetime, depth: int = TextElementType.BLOCK.value,
                 container: TextElement = None):
        super().__init__(lang=lang, text=text, depth=depth, container=container)
        self.parent_contribution = parent_contribution
        self.participant = participant
        self.timestamp = timestamp
        self.index = index

        # used for creating per-participant conversations, based on original contributions
        self.contribution_raw = contribution_raw

    def add_sentence(self, sentence: Sentence):
        self.components.append(sentence)
        sentence.container = self

    def get_parent(self) -> "Contribution":
        return self.parent_contribution

    def get_participant(self) -> Participant:
        return self.participant

    def get_timestamp(self) -> int:
        return self.timestamp

    def is_significant(self):
        return len(self.get_words()) >= SIGNIFICANT_LIMIT

    def get_raw_contribution(self) -> "Contribution":
        return self.contribution_raw
    
    def __eq__(self, other):
        if not isinstance(other, Contribution):
            return False
        return self.timestamp == other.timestamp and TextElement.__eq__(self, other)
    
    def __hash__(self):
        return hash((self.timestamp, TextElement.__hash__(self)))
    