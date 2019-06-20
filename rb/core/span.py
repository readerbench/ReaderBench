from typing import List

from rb.core.lang import Lang
from rb.core.text_element import TextElement
from rb.core.text_element_type import TextElementType
from rb.core.word import Word


class Span(TextElement):
    def __init__(self, lang: Lang, text: str, words: List[Word], depth: int = TextElementType.SPAN.value, container: TextElement=None):
        super().__init__(lang, text, depth, container=container)
        self.words = words

    def get_root(self) -> Word:
        return [word for word in self.words if word.head == word or word.head.i < self.words[0].i or word.head.i > self.words[-1].i][0]
