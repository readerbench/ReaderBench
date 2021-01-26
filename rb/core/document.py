from typing import List

from rb.core.block import Block
from rb.core.lang import Lang
from rb.core.sentence import Sentence
from rb.core.text_element import TextElement
from rb.core.text_element_type import TextElementType
from rb.core.word import Word
from rb.similarity.vector_model import (CorporaEnum, VectorModel,
                                        VectorModelType)
from rb.utils.downloader import download_tags


class Document(TextElement):
    

    def __init__(self, lang: Lang, text: str,
                 depth: int = TextElementType.DOC.value,
                 container: TextElement = None):
        TextElement.__init__(self, lang=lang, text=text,
                             depth=depth, container=container)
        text = text.replace("\n\n", "\n")
        for block in text.split("\n"):
            self.components.append(Block(lang=lang, text=block.strip(),
                                         container=self))
        count = 0
        for block in self.components:
            words = block.get_words()
            for word in words:
                word.index_in_doc += count
            count += len(words)

    def get_words(self) -> List[Word]:
        return [word for block in self.components for sent in block.components for word in sent.components]

    def get_sentences(self) -> List[Sentence]:
        return [sent for block in self.components for sent in block.components]

    def get_blocks(self) -> List[Block]:
        return self.components
