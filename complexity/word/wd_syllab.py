import pyphen

from rb.complexity.complexity_index import ComplexityIndex
from rb.complexity.index_category import IndexCategory
from rb.complexity.measure_function import MeasureFunction
from rb.core.lang import Lang
from rb.core.text_element import TextElement
from rb.utils.rblogger import Logger

logger = Logger.get_logger()


class WdSyllab(ComplexityIndex):
    
    def __init__(self, lang: Lang,
            reduce_depth: int, reduce_function: MeasureFunction):

        ComplexityIndex.__init__(self, lang=lang, category=IndexCategory.WORD,
                                 abbr="Syllab", reduce_depth=reduce_depth,
                                 reduce_function=reduce_function)
        self.pyphen = pyphen.Pyphen(lang=lang.name.lower())
        
    def _compute_value(self, element: TextElement) -> int:
        return len(self.pyphen.inserted(element.text).split('-'))
    
    def __getstate__(self):
        return {
            key: value
            for key, value in self.__dict__.items()
            if key != "pyphen"
        }