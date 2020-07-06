from rb.complexity.complexity_index import ComplexityIndex
from rb.complexity.index_category import IndexCategory
from rb.complexity.measure_function import MeasureFunction
from rb.complexity.rhythm.syllabified_dict import SyllabifiedDict
from rb.core.lang import Lang
from rb.core.text_element import TextElement
from rb.utils.rblogger import Logger

from typing import List

logger = Logger.get_logger()


class NoAlliterations(ComplexityIndex):

    def __init__(self, lang: Lang, reduce_depth: int,
                 reduce_function: MeasureFunction):

        ComplexityIndex.__init__(self, lang=lang, category=IndexCategory.RHYTHM,
                                 abbr="RhNoAlliterations", reduce_depth=reduce_depth,
                                 reduce_function=reduce_function)

        self.syllabified_dict = SyllabifiedDict.get_instance(lang)

    def process(self, element: TextElement) -> float:
        pass
