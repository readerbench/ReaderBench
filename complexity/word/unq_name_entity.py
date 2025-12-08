from rb.complexity.complexity_index import ComplexityIndex
from rb.core.lang import Lang
from rb.core.text_element import TextElement
from rb.complexity.index_category import IndexCategory
from rb.complexity.measure_function import MeasureFunction
from rb.complexity.word.name_entity_enum import NamedEntityONEnum 
from rb.core.text_element_type import TextElementType
from typing import List, Callable, Set
from rb.utils.rblogger import Logger

logger = Logger.get_logger()


class NoUnqNamedEntity(ComplexityIndex):

    """named_ent_type depends on the model, there are 2 enum options
        see name_entity_enum.py for those"""
    def __init__(self, lang: Lang, named_ent_type: NamedEntityONEnum,
            reduce_depth: int, reduce_function: MeasureFunction):

        ComplexityIndex.__init__(self, lang=lang, category=IndexCategory.WORD,
                                 abbr="UnqNmdEnt", reduce_depth=reduce_depth,
                                 reduce_function=reduce_function)
        self.named_ent_type = named_ent_type

    def _compute_value(self, element: TextElement) -> int:
        return len({word.text for word in element.get_words() if word.ent_type_ != 0 and word.ent_type_ == self.named_ent_type.name})
    
    def __repr__(self):
        return f"{self.reduce_function_abbr}({self.abbr}_{self.named_ent_type.name.lower()} / {self.reduce_depth_abbr})"