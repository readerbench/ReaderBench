from rb.complexity.complexity_index import ComplexityIndex
from rb.core.lang import Lang
from rb.core.text_element import TextElement
from rb.complexity.index_category import IndexCategory
from rb.complexity.measure_function import MeasureFunction
from rb.core.text_element_type import TextElementType
from rb.similarity.wordnet import get_all_paths_lengths_to_root
from typing import List, Callable
from rb.utils.rblogger import Logger

logger = Logger.get_logger()


class WdMaxDpthHypTree(ComplexityIndex):
    
    def __init__(self, lang: Lang, reduce_depth: int,
            reduce_function: MeasureFunction):

        ComplexityIndex.__init__(self, lang=lang, category=IndexCategory.WORD,
                                 abbr="MaxDepthHypTree", reduce_depth=reduce_depth,
                                 reduce_function=reduce_function)

    def _compute_value(self, element: TextElement) -> int:
        if element.is_content_word():
            paths = get_all_paths_lengths_to_root(element)
            return max(paths) if paths else 0
        else:
            return 0