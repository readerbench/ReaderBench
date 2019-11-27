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

logger = Logger.get_logger()


class ChEntropy(ComplexityIndex):
    

    def __init__(self, lang: Lang, reduce_depth: int,
                 reduce_function: MeasureFunction):

        ComplexityIndex.__init__(self, lang=lang, category=IndexCategory.SURFACE,
                                 abbr="ChEntropy", reduce_depth=reduce_depth,
                                 reduce_function=reduce_function)

    def process(self, element: TextElement) -> float:
        return self.reduce_function(self.compute_above(element))
    
    def compute_below(self, element: TextElement) -> List[float]:
        if element.is_word() == True:
            chars = list(element.text)
            counter = Counter(chars)
            nr_total_chars = len(chars)
            res = 0
            for _, v in counter.items():
                v = v / nr_total_chars
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