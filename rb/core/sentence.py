from typing import List, Tuple

from rb.core.lang import Lang
from rb.core.span import Span
from rb.core.text_element import TextElement
from rb.core.text_element_type import TextElementType
from rb.core.word import Word
from rb.parser.spacy_parser import SpacyParser
from spacy.tokens.doc import Doc

Dependency = Tuple[Word, Word, str]
Dependencies = List[Dependency]

class Sentence(TextElement):


    def __init__(self, lang: Lang, text: str,
                 depth: int = TextElementType.SENT.value,
                 container: TextElement = None):

        TextElement.__init__(self, lang=lang, text=text,
                             depth=depth, container=container)
        doc = SpacyParser.get_instance().parse(text, lang.value)

        for token in doc:
            word = Word(lang, token, container=self)
            self.components.append(word)
        for word, token in zip(self.components, doc):
            word.head = self.components[token.head.i]
            if word.head is not word:
                word.head.children.append(word)
        self.entities = [Span(lang, text=ent.text, words=[self.components[token.i] for token in ent])
                         for ent in doc.ents]
      
        self.root = self.components[list(doc.sents)[0].root.i]

    def get_dependencies(self) -> Dependencies:
        return [(word.head, word, word.dep) 
                for word in self.components 
                if word.head != word]
            
    def get_sentences(self) -> List["Sentence"]:
        return [self]

    def __str__(self):
        return self.text
    