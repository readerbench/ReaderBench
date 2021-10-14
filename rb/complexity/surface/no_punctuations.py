
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

class NoPunctuations(ComplexityIndex):
    

    def __init__(self, lang: Lang, reduce_depth: int,
                 reduce_function: MeasureFunction):

        ComplexityIndex.__init__(self, lang=lang, category=IndexCategory.SURFACE, 
                                 abbr="Punct", reduce_depth=reduce_depth,
                                 reduce_function=reduce_function)

    def _compute_value(self, element: TextElement) -> int:
        return sum(1 for word in element.get_words() if word.pos is POS.PUNCT)