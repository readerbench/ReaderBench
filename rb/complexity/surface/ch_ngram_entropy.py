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
                                 abbr="NgramEntr", reduce_depth=reduce_depth,
                                 reduce_function=reduce_function)
        self.ngram_size = ngram_size

    def get_ngrams(self, chars: str) -> List[str]:
        ngrams = []
        for i in range(len(chars) - (self.ngram_size.value - 1)):
            ngrams.append(chars[i:(i + self.ngram_size.value)])
        return ngrams

    def _compute_value(self, element: TextElement) -> float:
        if element.is_word() == True:
            chars = element.text
            ngrams = self.get_ngrams(chars)
            counter = Counter(ngrams)
            n = len(ngrams)
            return sum(- v / n * math.log(v / n) for v in counter.values())
    
    def __repr__(self):
        return f"{self.reduce_function_abbr}({self.abbr}_{self.ngram_size.value} / {self.reduce_depth_abbr})"