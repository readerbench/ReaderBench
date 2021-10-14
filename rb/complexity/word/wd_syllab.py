from rb.complexity.complexity_index import ComplexityIndex
from rb.core.lang import Lang
from rb.core.text_element import TextElement
from rb.complexity.index_category import IndexCategory
from rb.complexity.measure_function import MeasureFunction
from rb.core.text_element_type import TextElementType   
from typing import List, Callable
import pyphen

from rb.utils.rblogger import Logger

logger = Logger.get_logger()


class WdSyllab(ComplexityIndex):
    
    def __init__(self, lang: Lang,
            reduce_depth: int, reduce_function: MeasureFunction):

        ComplexityIndex.__init__(self, lang=lang, category=IndexCategory.WORD,
                                 abbr="Syllab", reduce_depth=reduce_depth,
                                 reduce_function=reduce_function)
        if lang is lang.RO:
            self.pyphen = pyphen.Pyphen(lang='ro')
        elif lang is lang.EN:
            self.pyphen = pyphen.Pyphen(lang='en')
        elif lang is lang.ES:
            self.pyphen = pyphen.Pyphen(lang='es')
        elif lang is lang.FR:
            self.pyphen = pyphen.Pyphen(lang='fr')
        elif lang is lang.DE:
             self.pyphen = pyphen.Pyphen(lang='de')
        elif lang is lang.IT:
             self.pyphen = pyphen.Pyphen(lang='it')
        elif lang is lang.NL:
             self.pyphen = pyphen.Pyphen(lang='nl')
        elif lang is lang.RU:
             self.pyphen = pyphen.Pyphen(lang='ru')
    
    def _compute_value(self, element: TextElement) -> int:
        return len(self.pyphen.inserted(element.text).split('-'))
    
    def __getstate__(self):
        return {
            key: value
            for key, value in self.__dict__.items()
            if key != "pyphen"
        }