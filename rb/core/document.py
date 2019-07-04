from rb.core.lang import Lang
from rb.core.text_element import TextElement
from rb.core.block import Block
from rb.core.text_element_type import TextElementType
from rb.core.word import Word
from typing import List

class Document(TextElement):
    

    def __init__(self, lang: Lang, text: str,
                 depth: int = TextElementType.DOC.value,
                 container: TextElement = None):
        TextElement.__init__(self, lang=lang, text=text,
                             depth=depth, container=container)
        for block in text.split("\n"):
            self.components.append(Block(lang=lang, text=block.strip(),
                                         container=self))

    def get_tokens(self) -> List[Word]:
        tokens = []
        for paragraph in self.components:
            for sentence in paragraph.components:
                for token in sentence.components:
                    tokens.append(token)
        return tokens

    def __str__(self):
        return self.text
    