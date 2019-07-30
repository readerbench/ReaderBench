from typing import List

from rb.core.lang import Lang
from rb.complexity.measure_function import MeasureFunction
from rb.core.text_element_type import TextElementType
from rb.complexity.measure_function import MeasureFunction

# create all indices
# dependencies need to be putted in function because otherwise circular dependencies happens
def create(lang: Lang) -> List["ComplexityIndex"]:
    from rb.complexity.rhythm.syllab import Syllab

    indices = []
    indices.append(Syllab(lang, TextElementType.WORD.value, MeasureFunction.AVG))
    return indices