from rb.core.lang import Lang
from rb.core.text_element import TextElement
from rb.complexity.complexity_index import ComplexityIndex
from rb.complexity.index_category import IndexCategory
from rb.complexity.measure_function import MeasureFunction


class RhythmicIndex(ComplexityIndex):

    def __init__(self, lang: Lang, reduce_depth: int,
                 reduce_function: MeasureFunction):
        ComplexityIndex.__init__(self, lang=lang, category=IndexCategory.RHYTHM,
                                 abbr="RhIndex", reduce_depth=reduce_depth,
                                 reduce_function=reduce_function)

    def process(self, element: TextElement) -> float:
        pass
