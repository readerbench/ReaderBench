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


class WdEntropy(ComplexityIndex):
    
    def __init__(self, lang: Lang, reduce_depth: int,
                 reduce_function: MeasureFunction):

        ComplexityIndex.__init__(self, lang=lang, category=IndexCategory.SURFACE,
                                 abbr="WdEntr", reduce_depth=reduce_depth,
                                 reduce_function=reduce_function)

    def _compute_value(self, element: TextElement) -> float:
        lemmas = [word.lemma for word in element.get_words()]
        n = len(lemmas)
        counter = Counter(lemmas)
        return sum(- v / n * math.log(v / n) for v in counter.values())