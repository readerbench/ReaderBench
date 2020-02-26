from typing import List, Dict

import spacy

from rb.core.lang import Lang
from rb.core.text_element import TextElement
from rb.core.text_element_type import TextElementType
from rb.core.word import Word


class Span(TextElement):


    def __init__(self, lang: Lang, text: str, words: List[Word],
         depth: int = TextElementType.SPAN.value):
        super().__init__(lang, text, depth, container=words[0].container)
        self.components = words

    def get_root(self) -> Word:
        return [word for word in self.components 
                if word.head == word or 
                   word.head.index_in_doc < self.components[0].index_in_doc or 
                   word.head.index_in_doc > self.components[-1].index_in_doc
            ][0]
    
    @classmethod
    def from_spacy_span(cls, lang: Lang, spacy_span: spacy.tokens.Span, words: Dict[int, Word]) -> "Word":
        text = spacy_span.text
        our_words = [words[i] for i in range(spacy_span.start, spacy_span.end)]
        return Span(lang=lang, text=text, words=our_words)
