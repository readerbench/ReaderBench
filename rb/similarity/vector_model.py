import abc
import os
from enum import Enum
from typing import Dict, List, Tuple, Union

import numpy as np
from numpy.linalg import norm
from rb.core.lang import Lang
from rb.core.text_element import TextElement
from rb.core.word import Word
from rb.similarity.vector import Vector
from rb.utils.downloader import check_version, download_model
from rb.utils.rblogger import Logger

WordSimilarity = Tuple[str, float]

logger = Logger.get_logger()


class VectorModelType(Enum):
    LSA = 0
    LDA = 1
    WORD2VEC = 2
    FASTTEXT = 3
    GLOVE = 4
    TRANSFORMER = 5

    @classmethod
    def from_str(cls, label: str) -> "VectorModelType":
        try:
            return cls[label.upper()]
        except:
            pass
        if "2" in label:
            return cls.WORD2VEC
        return None


class CorporaEnum(Enum):
    README = 'readme'
    COCA = 'coca'
    JOSE_ANTONIO = 'jose_antonio'
    RNC_WIKIPEDIA = 'rnc_wikipedia'
    LE_MONDE = 'le_monde'
    WIKI = 'wiki'


class VectorModel:

    
    def __init__(self, type: VectorModelType, name: str, lang: Lang, size: int):
        self.lang = lang
        self.type = type
        self.name = name
        self.size = size
        
    def get_vector(self, elem: Union[str, TextElement]) -> Vector:
        if isinstance(elem, str):
            raise NotImplementedError
        return elem.vectors.get(self)
            
    
    def similarity(self, a: Union[TextElement, Vector], b: Union[TextElement, Vector]) -> float:
        if isinstance(a, TextElement) and isinstance(b, TextElement) and a == b:
            return 1.0
        if isinstance(a, TextElement):
            a = self.get_vector(a)
        if isinstance(b, TextElement):
            b = self.get_vector(b)
        if a is None or b is None:
            return 0.0
        return Vector.cosine_similarity(a, b)