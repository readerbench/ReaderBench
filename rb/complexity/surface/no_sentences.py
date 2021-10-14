from rb.core.lang import Lang
from rb.core.text_element import TextElement
from rb.core.word import Word
from rb.complexity.complexity_index import ComplexityIndex
from rb.complexity.index_category import IndexCategory
from rb.complexity.measure_function import MeasureFunction
from rb.core.text_element_type import TextElementType
from typing import Callable, List
from rb.utils.rblogger import Logger
logger = Logger.get_logger()


class NoSentences(ComplexityIndex):
    
    
    def __init__(self, lang: Lang, reduce_depth: int,
                 reduce_function: MeasureFunction):

        ComplexityIndex.__init__(self, lang=lang, category=IndexCategory.SURFACE, 
                                 abbr="Sent", reduce_depth=reduce_depth,
                                 reduce_function=reduce_function)

    def _compute_value(self, element: TextElement) -> int:
        return len(element.get_sentences())