from rb.complexity.complexity_index import ComplexityIndex
from rb.core.lang import Lang
from rb.core.text_element import TextElement
from rb.complexity.index_category import IndexCategory
from rb.complexity.measure_function import MeasureFunction
from rb.core.text_element_type import TextElementType   
from typing import List, Callable, Set
from rb.core.pos import POS as PosEum
from rb.utils.rblogger import Logger

logger = Logger.get_logger()


class UnqPosMain(ComplexityIndex):

    
    def __init__(self, lang: Lang, pos_type: PosEum,
            reduce_depth: int, reduce_function: MeasureFunction):

        ComplexityIndex.__init__(self, lang=lang, category=IndexCategory.MORPHOLOGY,
                                 abbr="UnqPOSMain", reduce_depth=reduce_depth,
                                 reduce_function=reduce_function)
        self.pos_type = pos_type

    def process(self, element: TextElement) -> float:
        return self.reduce_function(self.compute_above(element))

    def compute_below(self, element: TextElement) -> Set[str]:
        if element.is_word() == True:
            res = set()
            if element.pos.name == self.pos_type.name:
                res.add(element.text)
            return res
        elif element.depth <= self.reduce_depth:
            res = set()
            for child in element.components:
                res.update(self.compute_below(child))
            return res

    def compute_above(self, element: TextElement) -> List[float]:
        if element.depth > self.reduce_depth:
            values = []
            for child in element.components:
                values += self.compute_above(child)
            element.indices[self] = self.reduce_function(values)
        elif element.depth == self.reduce_depth:
            element.indices[self] = len(self.compute_below(element))
            values = [len(self.compute_below(element))]
            element.indices[self] = self.reduce_function(values)
        else:
            logger.error('wrong reduce depth value.')
        return values

    def __repr__(self):
        return self.reduce_function_abbr + self.reduce_depth_abbr + self.abbr + "_" + self.pos_type.name.lower()
