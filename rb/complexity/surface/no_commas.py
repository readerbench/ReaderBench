
from rb.core.lang import Lang
from rb.core.text_element import TextElement
from rb.complexity.complexity_index import ComplexityIndex
from rb.complexity.index_category import IndexCategory
from rb.complexity.measure_function import MeasureFunction
from rb.core.text_element_type import TextElementType
from rb.core.pos import POS
from typing import Callable, List
from rb.utils.rblogger import Logger

logger = Logger.get_logger()

class NoCommas(ComplexityIndex):
    
    
    def __init__(self, lang: Lang, reduce_depth: int, 
                 reduce_function: MeasureFunction):

        ComplexityIndex.__init__(self, lang=lang, category=IndexCategory.SURFACE,
                                 abbr="Commas", reduce_depth=reduce_depth, 
                                 reduce_function=reduce_function)

    def process(self, element: TextElement) -> float:
        x = self.reduce_function(self.compute_above(element))
        return x

    def compute_below(self, element: TextElement) -> float:
        if element.is_sentence() == True:
            res = sum(1 for word in element.components if word.text == ",")
            return res
        elif element.depth <= self.reduce_depth:
            res = 0
            for child in element.components:
                res += self.compute_below(child)
            return res
    
    def compute_above(self, element: TextElement) -> List[float]:
        if element.depth > self.reduce_depth:
            values = []
            for child in element.components:
                values += self.compute_above(child)
            element.indices[self] = self.reduce_function(values)
        elif element.depth == self.reduce_depth:
            values = [self.compute_below(element)]
            element.indices[self] = self.reduce_function(values)
        else:
            logger.error('wrong reduce depth value.')
        return values