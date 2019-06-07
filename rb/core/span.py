from typing import List

from rb.core.lang import Lang
from rb.core.text_element import TextElement
from rb.core.text_element_type import TextElementType
from rb.core.word import Word


class Span(TextElement):
    def __init__(self, lang: Lang, text: str, words: List[Word], depth: int = TextElementType.SPAN.value, container: TextElement=None):
        super().__init__(lang, text, depth, container=container)
        self.words = words
