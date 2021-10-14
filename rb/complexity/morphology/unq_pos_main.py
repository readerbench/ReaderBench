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
                                 abbr="UnqPOS", reduce_depth=reduce_depth,
                                 reduce_function=reduce_function)
        self.pos_type = pos_type

    def _compute_value(self, element: TextElement) -> int:
        return len({word.text for word in element.get_words() if word.pos is self.pos_type})

    def __repr__(self):
        return f"{self.reduce_function_abbr}({self.abbr}_{self.pos_type.name.lower()} / {self.reduce_depth_abbr})"
