from rb.complexity.complexity_index import ComplexityIndex
from rb.complexity.syntax.dep_enum import DepEnum
from rb.core.lang import Lang
from rb.core.text_element import TextElement
from rb.complexity.index_category import IndexCategory
from rb.complexity.measure_function import MeasureFunction
from rb.core.text_element_type import TextElementType   
from typing import List, Callable
from rb.utils.rblogger import Logger

logger = Logger.get_logger()


class DepIndex(ComplexityIndex):

    
    def __init__(self, lang: Lang, dep_type: DepEnum,
         reduce_depth: int, reduce_function: MeasureFunction):

        ComplexityIndex.__init__(self, lang=lang, category=IndexCategory.SYNTAX,
                                 abbr="Dep", reduce_depth=reduce_depth,
                                 reduce_function=reduce_function)
        self.dep_type = dep_type

    def process(self, element: TextElement) -> float:
        return self.reduce_function(self.compute_above(element))

    def compute_below(self, element: TextElement) -> float:
        if element.is_sentence() == True:
            res = sum(1 for word in element.components if word.dep == self.dep_type.name.lower())
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

    def __repr__(self):
        return self.reduce_function_abbr + self.reduce_depth_abbr + self.abbr + "_" + self.dep_type.name.lower()

