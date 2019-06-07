from enum import Enum
from typing import Union
from typing import List
from typing import Tuple
WordSimilarity = Tuple[str, float]

import numpy as np
from numpy.linalg import norm

from rb.core.lang import Lang
from rb.core.text_element import TextElement
from rb.core.word import Word

class VectorModelType(Enum):
    LSA = 0,
    LDA = 1,
    WORD2VEC = 2,
    FASTTEXT = 3,
    GLOVE = 4


class VectorModel:

    
    def __init__(self, type: VectorModelType, name: str, lang: Lang, size: int = 300):
        self.lang = lang
        self.type = type
        self.name = name
        self.size = size
        self.vectors = {}

    def get_vector(self, elem: Union[str, TextElement]) -> np.array:
        if isinstance(elem, str):
            return self.vectors[elem]
        if not self in elem.vectors:
            if isinstance(elem, Word):
                if elem.text in self.vectors:
                    elem.vectors[self] = self.vectors[elem.text]
                elif elem.lemma in self.vectors:
                    elem.vectors[self] = self.vectors[elem.lemma]
                else:
                    return None
            else:
                vectors = [self.get_vector(child) for child in elem.components]
                vectors = [v for v in vectors if v is not None]
                elem.vectors[self] = np.sum(vectors, axis=0) if len(vectors) > 0 else None
        return elem.vectors[self]
            
    
    def similarity(self, a: Union[TextElement, np.array], b: Union[TextElement, np.array]) -> float:
        if isinstance(a, TextElement) and isinstance(b, TextElement) and a == b:
            return 1.0
        if isinstance(a, TextElement):
            a = self.get_vector(a)
        if isinstance(b, TextElement):
            b = self.get_vector(b)
        if a is None or b is None:
            return 0.0
        return np.dot(a, b) / (norm(a) * norm(b))


    """
    Loads vectors from file.
    If this method is used the self.size will be overwriten.
    """
    def load_vectors_from_txt_file(self, filepath: str) -> None:
        with open(filepath, 'r', encoding='utf-8') as f:
            line = f.readline()
            line_split = line.split()
            no_of_words = int(line_split[0])
            no_of_dimensions = int(line_split[1])
            self.size = no_of_dimensions

            for _ in range(no_of_words):
                line = f.readline()
                line_split = line.split()
                word = ""
                dimensions = []
                for index, value in enumerate(line_split):
                    if index == 0:
                        word = value
                    else:
                        dimensions.append(float(value))
                self.vectors[word] = dimensions


    def most_similar(self, elem: Union[str, TextElement], topN: int = 10, threshold: float = None) -> List[WordSimilarity]:
        elem_vector = self.get_vector(elem)

        all_similarities = []
        for key, value in self.vectors.items():
            if key == elem:
                continue
            similarity = self.similarity(elem_vector, value)
            if not threshold or similarity > threshold:
                all_similarities.append((key, similarity))
        
        all_similarities.sort(key=lambda tup: tup[1], reverse=True)

        return all_similarities[:topN]