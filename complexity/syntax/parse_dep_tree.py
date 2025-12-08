from rb.complexity.complexity_index import ComplexityIndex
from rb.core.lang import Lang
from rb.core.text_element import TextElement
from rb.core.word import Word
from rb.complexity.index_category import IndexCategory
from rb.complexity.measure_function import MeasureFunction
from rb.core.text_element_type import TextElementType   
from typing import List, Callable
from rb.utils.rblogger import Logger

logger = Logger.get_logger()


class ParseDepTree(ComplexityIndex):

    
    def __init__(self, lang: Lang, reduce_depth: int,
            reduce_function: MeasureFunction):

        ComplexityIndex.__init__(self, lang=lang, category=IndexCategory.SYNTAX,
                                 abbr="ParseDepth", reduce_depth=reduce_depth,
                                 reduce_function=reduce_function)

    def _compute_value(self, element: TextElement) -> int:
        return self.compute_height(element.root)

    def compute_height(self, word: Word) -> int:
        if not list(word.children):
            return 1
        else:
            return 1 + max(self.compute_height(x) for x in word.children)