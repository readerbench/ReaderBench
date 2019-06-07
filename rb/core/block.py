from rb.parser.spacy_parser import SpacyParser

from rb.core.lang import Lang
from rb.core.text_element import TextElement
from rb.core.sentence import Sentence
from rb.core.text_element_type import TextElementType

class Block(TextElement):


    def __init__(self, lang: Lang, text: str,
                 depth: int = TextElementType.BLOCK.value,
                 container: TextElement = None):

        TextElement.__init__(self, lang=lang, text=text,
                             depth=depth, container=container)
        for sentence in SpacyParser.get_instance().tokenize_sentences(text):
            self.components.append(Sentence(lang, sentence, container=self))

    def __str__(self):
        return self.text
    