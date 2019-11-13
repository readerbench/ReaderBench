from rb.core.lang import Lang
from rb.core.text_element import TextElement
from rb.core.word import Word
from rb.complexity.complexity_index import ComplexityIndex
from rb.complexity.index_category import IndexCategory
from rb.core.text_element_type import TextElementType
from rb.complexity.measure_function import MeasureFunction
from typing import Callable, List
from collections import Counter
from rb.utils.rblogger import Logger
import math
from enum import Enum

logger = Logger.get_logger()


class ChNgramEntropyEnum(Enum):
    ONE = 1
    TWO = 2
    THREE = 3


class ChNgramEntropy(ComplexityIndex):
    def __init__(self, lang: Lang, ngram_size: ChNgramEntropyEnum,
        reduce_depth: int, reduce_function: MeasureFunction):

        ComplexityIndex.__init__(self, lang=lang, category=IndexCategory.SURFACE,
                                 abbr="ChNgramEntropy", reduce_depth=reduce_depth,
                                 reduce_function=reduce_function)
        self.ngram_size = ngram_size

    def process(self, element: TextElement) -> float:
        return self.reduce_function(self.compute_above(element))
    
    def get_ngrams(self, chars: List[str]) -> List[str]:
        ngrams = []
        for i in range(len(chars) - (self.ngram_size.value - 1)):
            values = []
            for j in range(0, self.ngram_size.value):
               values.append(chars[i + j])
            ngrams.append("".join(values))
        return ngrams

    def compute_below(self, element: TextElement) -> List[float]:
        if element.is_word() == True:
            chars = list(element.text)
            ngrams = self.get_ngrams(chars)
            counter = Counter(ngrams)
            nr_total_ngrams = len(ngrams)
            res = 0
            for _, v in counter.items():
                v = v / nr_total_ngrams
                res += -v * math.log(v)
            return [res]
        elif element.depth <= self.reduce_depth:
            res = []
            for child in element.components:
                res += self.compute_below(child)
            return res
        else:
            logger.error('wrong reduce depth value.')
    
    def compute_above(self, element: TextElement) -> List[float]:
        if element.depth > self.reduce_depth:
            values = []
            for child in element.components:
                values += self.compute_above(child)
            element.indices[self] = self.reduce_function(values)
        elif element.depth == self.reduce_depth:
            values = self.compute_below(element)
            element.indices[self] = self.reduce_function(values)
        else:
            logger.error('wrong reduce depth value.')
        return values
    
    def __repr__(self):
        return self.abbr + "_" + str(self.ngram_size.value)