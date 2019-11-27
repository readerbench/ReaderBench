from typing import List

from rb.core.lang import Lang
from rb.complexity.measure_function import MeasureFunction
from rb.core.text_element_type import TextElementType
from rb.complexity.measure_function import MeasureFunction
from rb.cna.cna_graph import CnaGraph

# create all indices
# dependencies need to be putted in function because otherwise circular dependencies happens
def create(lang: Lang, cna_graph: CnaGraph) -> List["ComplexityIndex"]:
    from rb.complexity.syntax.parse_dep_tree import ParseDepTree
    from rb.complexity.syntax.dep import DepIndex
    from rb.complexity.syntax.dep_enum import DepEnum

    indices = []
    indices.append(ParseDepTree(lang, TextElementType.SENT.value, MeasureFunction.AVG))
    indices.append(ParseDepTree(lang, TextElementType.SENT.value, MeasureFunction.STDEV))

    indices.append(ParseDepTree(lang, TextElementType.BLOCK.value, MeasureFunction.AVG))
    indices.append(ParseDepTree(lang, TextElementType.BLOCK.value, MeasureFunction.STDEV))

    for dep_type in DepEnum:
        indices.append(DepIndex(lang, dep_type, TextElementType.SENT.value, MeasureFunction.AVG))
        indices.append(DepIndex(lang, dep_type, TextElementType.BLOCK.value, MeasureFunction.AVG))
    return indices
