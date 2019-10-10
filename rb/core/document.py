from rb.core.lang import Lang
from rb.core.text_element import TextElement
from rb.core.block import Block
from rb.core.sentence import Sentence
from rb.core.text_element_type import TextElementType
from rb.core.word import Word
from rb.similarity.vector_model import VectorModelType, CorporaEnum, VectorModel
from rb.similarity.vector_model_instance import VECTOR_MODELS
from rb.similarity.vector_model import VectorModelType, CorporaEnum, VectorModel


from rb.utils.downloader import download_tags 
from typing import List

class Document(TextElement):
    

    def __init__(self, lang: Lang, text: str,
                 vector_model: VectorModel = None,
                 depth: int = TextElementType.DOC.value,
                 container: TextElement = None):
        from rb.cna.cna_graph import CnaGraph
        TextElement.__init__(self, lang=lang, text=text,
                             depth=depth, container=container)
        text = text.replace("\n\n", "\n")
        for block in text.split("\n"):
            self.components.append(Block(lang=lang, text=block.strip(),
                                         container=self))

    def get_words(self) -> List[Word]:
        return [word for block in self.components for sent in block.components for word in sent.components]

    def get_sentences(self) -> List[Sentence]:
        return [sent for block in self.components for sent in block.components]

    def get_blocks(self) -> List[Block]:
        return self.components

    def __str__(self):
        return self.text
    