from typing import List

from rb.core.lang import Lang
from rb.complexity.measure_function import MeasureFunction
from rb.core.text_element_type import TextElementType
from rb.complexity.measure_function import MeasureFunction

# create all indices
# dependencies need to be putted in function because otherwise circular dependencies happens
def create(lang: Lang) -> List["ComplexityIndex"]:
    from rb.complexity.syntax.dep import DepIndex
    from rb.complexity.syntax.dep_enum import DepEnum
    from rb.complexity.syntax.parse_dep_tree import ParseDepTree

    indices = []
    indices.append(ParseDepTree(lang, TextElementType.SENT.value, MeasureFunction.AVG))
    #indices = [DepIndex(lang, dep, TextElementType.BLOCK, MeasureFunction.AVG) for dep in DepEnum]
    #indices += [DepIndex(lang, dep, TextElementType.BLOCK, MeasureFunction.STDEV) for dep in DepEnum]
    return indices
