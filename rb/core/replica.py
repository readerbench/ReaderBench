from typing import List

from rb.core.lang import Lang
from rb.core.pos import POS
from rb.core.text_element import TextElement
from rb.core.text_element_type import TextElementType
from rb.parser.spacy_parser import SpacyParser
from rb.core.pos_features.pos_feature_extractor import POSFeatureExtractor
from spacy.tokens import Token


class Replica(TextElement):

	Replica parent
	Replica 

	def __init__(self, lang: Lang, text: str,
                 depth: int = TextElementType.BLOCK.value,
                 container: TextElement = None,
                 parent_replica: Replica,
                 timestamp: int):

		TextElement.__init__(self, lang=lang, text=text,
                             depth=depth, container=container)
		self.parent_replica = parent_replica
		self.timestamp = timestamp

	def get_words()

	def get_sentences(self) -> List[Sentence]:
        return self.components

    def __str__(self):
        return self.text






