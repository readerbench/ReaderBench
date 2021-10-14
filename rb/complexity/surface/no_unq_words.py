
from rb.core.lang import Lang
from rb.core.text_element import TextElement
from rb.complexity.complexity_index import ComplexityIndex
from rb.complexity.index_category import IndexCategory
from rb.complexity.measure_function import MeasureFunction
from typing import Set
from rb.core.text_element_type import TextElementType
from typing import Callable, List, Set
from rb.utils.rblogger import Logger

logger = Logger.get_logger()

class NoUniqueWordsIndex(ComplexityIndex):
    

    def __init__(self, lang: Lang, reduce_depth: int,
                 reduce_function: MeasureFunction):

        ComplexityIndex.__init__(self, lang=lang, category=IndexCategory.SURFACE,
                                 abbr="UnqWd", reduce_depth=reduce_depth,
                                 reduce_function=reduce_function)

    def _compute_value(self, element: TextElement) -> int:
        return len({word.text for word in element.get_words()})