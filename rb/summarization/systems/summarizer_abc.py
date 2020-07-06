from abc import ABC, abstractmethod
from typing import Iterable

from rb.core.lang import Lang
from rb.parser.spacy_parser import SpacyParser
from rb.summarization.utils.parser.custom_parser import CustomParser


class Summarizer(ABC):

    def __init__(self):
        ABC.__init__(self)
        self._parser = CustomParser.get_instance()

    @property
    def parser(self):
        return self._parser

    @abstractmethod
    def summarize(self, doc: str, lang: Lang, parser: SpacyParser, ratio: float, word_count: int) -> Iterable[str]:
        pass
