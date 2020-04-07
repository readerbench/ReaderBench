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

    def process(self, element: TextElement) -> float:
        return self.reduce_function(self.compute_above(element))

    def compute_below(self, element: TextElement) -> Set[str]:
        if element.is_word() == True:
            if element.ent_type_ != 0 and element.ent_type_ == self.named_ent_type.name:
                return set([element.text])
            else:
                return set()
        elif element.depth <= self.reduce_depth:
            res = 0
            for child in element.components:
                res += self.compute_below(child)
            return res
    
    def compute_above(self, element: TextElement) -> List[float]:
        if element.depth > self.reduce_depth:
            values = []
            for child in element.components:
                values += self.compute_above(child)
            element.indices[self] = self.reduce_function(values)
        elif element.depth == self.reduce_depth:
            values = [self.compute_below(element)]
            element.indices[self] = self.reduce_function(values)
        else:
            logger.error('wrong reduce depth value.')
        return values
    
    def __repr__(self):
        return self.reduce_function_abbr + self.reduce_depth_abbr + self.abbr + "_" + self.named_ent_type.name.lower()