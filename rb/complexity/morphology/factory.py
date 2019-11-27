from typing import List

from rb.core.lang import Lang
from rb.complexity.measure_function import MeasureFunction
from rb.core.text_element_type import TextElementType
from rb.complexity.measure_function import MeasureFunction
from rb.core.pos import POS as PosEum
from rb.cna.cna_graph import CnaGraph


# create all indices
# dependencies need to be putted in function because otherwise circular dependencies happens
def create(lang: Lang, cna_graph: CnaGraph) -> List["ComplexityIndex"]:
    from rb.complexity.syntax.dep_enum import DepEnum
    from rb.complexity.morphology.pos_main import PosMain
    from rb.complexity.morphology.unq_pos_main import UnqPosMain
    from rb.complexity.morphology.pron_type_enum import PronounTypeEnum
    from rb.complexity.morphology.pron import Pronoun
    
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


    indices.append(PosMain(lang, PosEum.NOUN, TextElementType.BLOCK.value, MeasureFunction.AVG))
    indices.append(PosMain(lang, PosEum.VERB, TextElementType.BLOCK.value, MeasureFunction.AVG))
    indices.append(PosMain(lang, PosEum.ADJ, TextElementType.BLOCK.value, MeasureFunction.AVG))
    indices.append(PosMain(lang, PosEum.ADV, TextElementType.BLOCK.value, MeasureFunction.AVG))
    indices.append(PosMain(lang, PosEum.PRON, TextElementType.BLOCK.value, MeasureFunction.AVG))

    indices.append(UnqPosMain(lang, PosEum.NOUN, TextElementType.BLOCK.value, MeasureFunction.AVG))
    indices.append(UnqPosMain(lang, PosEum.VERB, TextElementType.BLOCK.value, MeasureFunction.AVG))
    indices.append(UnqPosMain(lang, PosEum.ADJ, TextElementType.BLOCK.value, MeasureFunction.AVG))
    indices.append(UnqPosMain(lang, PosEum.ADV, TextElementType.BLOCK.value, MeasureFunction.AVG))
    indices.append(UnqPosMain(lang, PosEum.PRON, TextElementType.BLOCK.value, MeasureFunction.AVG))
    
    if lang is Lang.RO or lang is Lang.EN:
        for pt in PronounTypeEnum:
            indices.append(Pronoun(lang, pt, TextElementType.SENT.value, MeasureFunction.AVG))
            indices.append(Pronoun(lang, pt, TextElementType.BLOCK.value, MeasureFunction.AVG))

    return indices