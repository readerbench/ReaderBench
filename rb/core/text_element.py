
from rb.core.lang import Lang
from rb.core.text_element_type import TextElementType
import numpy as np
from typing import List, Dict

from rb.core.lang import Lang


class TextElement:
    """General class of TextElement
    Each node from the document's tree is a TextElement

    Attributes
    ----------
    lang : Lang
        language where the index is applied
    text : str
        raw text of the element
    depth : int
        Depth in the document's tree. It is used to determine what type of element it is. 
        depth 0 means is a word, depth 1 means sentence
        It can increases as much as you want, the last elements
        in the tree being always of the type Document
    container: parent textelement
    """

    def __init__(self, lang: Lang, text: str,
                 depth: int,
                 container: 'TextElement' = None):
        from rb.complexity.complexity_index import ComplexityIndex

        self.text = None
        if text != None:
            self.text = text.strip()

        self.lang = lang
        self.container = container
        self.vectors = {}
        self.components: List[TextElement] = []
        self.vectors_initialized = False
        self.indices: Dict[ComplexityIndex, float] = {}
        self.depth = depth
    
    def is_community(self) -> bool:
        return self.depth >= TextElementType.COMM.value

    def is_conversation(self) -> bool:
        return self.depth >= TextElementType.CONV.value

    def is_contribution(self) -> bool:
        return self.depth == TextElementType.DOC.value

    def is_document(self) -> bool:
        return self.depth >= TextElementType.DOC.value

    def is_block(self) -> bool:
        return self.depth ==  TextElementType.BLOCK.value
    
    def is_sentence(self) -> bool:
        return self.depth == TextElementType.SENT.value
    
    def is_word(self) -> bool:
        return self.depth == TextElementType.WORD.value
        
    def get_parent_document(self) -> "Document":
        if self.is_document():
            return self
        else:
            return self.container.get_parent_document()

    def get_vector(self, model: 'VectorModel') -> np.array:
        return self.vectors[model]

    def get_words(self) -> List["Word"]:
        if self.depth == TextElementType.WORD.value:
            return [self]
        return [word for child in self.components for word in child.get_words()]

    def __eq__(self, other):
        if isinstance(other, TextElement):
            return self.depth == self.depth and self.text == other.text
        return NotImplemented

    def __hash__(self):
        return hash((self.depth, self.text))

    def __iter__(self):
        return iter(self.components)

    def __str__(self):
        return self.text

    def __getstate__(self):
        return {
            key: value
            for key, value in self.__dict__.items()
            if key != "cna_graph"
        }
        
    def __setstate__(self, state):
        for key, value in state.items():
            setattr(self, key, value)

    def get_sentences(self) -> List["Sentence"]:
        if self.depth == TextElementType.SENT.value:
            return [self]

        return  [sent for child in self.components for sent in child.get_sentences()]
