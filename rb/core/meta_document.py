from typing import List, Union

from rb.core.block import Block
from rb.core.lang import Lang
from rb.core.sentence import Sentence
from rb.core.text_element import TextElement
from rb.core.text_element_type import TextElementType
from rb.core.word import Word
from rb.similarity.vector_model import (CorporaEnum, VectorModel,
                                        VectorModelType)
from rb.utils.downloader import download_tags
from rb.core.document import Document

class MetaDocument(TextElement):
    

    def __init__(self, lang: Lang, text: Union[str, List[str]],
                 depth: int = TextElementType.CONV.value,
                 container: TextElement = None):
        if isinstance(text, str):
            sections = text.split("\n\n")
        else:
            sections = text
            text = "\n\n".join(text)
        TextElement.__init__(self, lang=lang, text=text,
                             depth=depth, container=container)
        self.components = [Document(self.lang, section, container=self) for section in sections]
        