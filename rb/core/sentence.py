from typing import List, Tuple, Union

import spacy
from rb.core.lang import Lang
from rb.core.span import Span
from rb.core.text_element import TextElement
from rb.core.text_element_type import TextElementType
from rb.core.word import Word
from rb.parser.spacy_parser import SpacyParser
from spacy.tokens import Doc

Dependency = Tuple[Word, Word, str]
Dependencies = List[Dependency]


class Sentence(TextElement):


    def __init__(self, lang: Lang, text: Union[spacy.tokens.Span, str],
                 depth: int = TextElementType.SENT.value,
                 container: TextElement = None):

        TextElement.__init__(self, lang=lang, text=text if isinstance(text, str) else text.text,
                             depth=depth, container=container)
        if isinstance(text, str):
            text = SpacyParser.get_instance().parse(text, lang)

        words = {token.i: Word(lang, token, container=self) for token in text}
        for word, token in zip(words.values(), text):
            word.head = words[token.head.i]
            if word.head is not word:
                word.head.children.append(word)
        self.entities = [Span(lang, text=ent.text, words=[words[token.i] for token in ent])
                         for ent in text.ents]
        if isinstance(text, Doc):
            self.root = words[[sent for sent in text.sents][0].root.i]
        else:
            self.root = words[text.root.i]
        self.components = [word for word in words.values()]

    def get_dependencies(self) -> Dependencies:
        return [(word.head, word, word.dep) 
                for word in self.components 
                if word.head != word]
            
    def get_sentences(self) -> List["Sentence"]:
        return [self]
    
    def get_words(self) -> List[Word]:
        return self.components