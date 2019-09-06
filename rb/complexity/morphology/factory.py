from typing import List

from rb.core.lang import Lang
from rb.complexity.measure_function import MeasureFunction
from rb.core.text_element_type import TextElementType
from rb.complexity.measure_function import MeasureFunction
from rb.core.pos import POS as PosEum

# create all indices
# dependencies need to be putted in function because otherwise circular dependencies happens
def create(lang: Lang) -> List["ComplexityIndex"]:
    from rb.complexity.syntax.dep_enum import DepEnum
    from rb.complexity.morphology.pos_main import PosMain
    from rb.complexity.morphology.unq_pos_main import UnqPosMain

    indices = []
    indices.append(PosMain(lang, PosEum.NOUN, TextElementType.SENT.value, MeasureFunction.AVG))
    indices.append(PosMain(lang, PosEum.VERB, TextElementType.SENT.value, MeasureFunction.AVG))
    indices.append(PosMain(lang, PosEum.ADJ, TextElementType.SENT.value, MeasureFunction.AVG))
    indices.append(PosMain(lang, PosEum.ADV, TextElementType.SENT.value, MeasureFunction.AVG))
    indices.append(PosMain(lang, PosEum.PRON, TextElementType.SENT.value, MeasureFunction.AVG))

    indices.append(UnqPosMain(lang, PosEum.NOUN, TextElementType.SENT.value, MeasureFunction.AVG))
    indices.append(UnqPosMain(lang, PosEum.VERB, TextElementType.SENT.value, MeasureFunction.AVG))
    indices.append(UnqPosMain(lang, PosEum.ADJ, TextElementType.SENT.value, MeasureFunction.AVG))
    indices.append(UnqPosMain(lang, PosEum.ADV, TextElementType.SENT.value, MeasureFunction.AVG))
    indices.append(UnqPosMain(lang, PosEum.PRON, TextElementType.SENT.value, MeasureFunction.AVG))
    
    return indices