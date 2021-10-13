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
    
    text_element_types = [TextElementType.SENT.value, TextElementType.BLOCK.value, TextElementType.DOC.value]
    measure_functions = [MeasureFunction.AVG, MeasureFunction.STDEV, MeasureFunction.MAX]
    parts_of_speech = [PosEum.NOUN, PosEum.VERB, PosEum.ADJ, PosEum.ADV, PosEum.PRON]
    
    for text_element in text_element_types:
        for pos in parts_of_speech:
            for measure_function in measure_functions:
                indices.append(PosMain(lang, pos, text_element, measure_function))
                indices.append(UnqPosMain(lang, pos, text_element, measure_function))
    
    if lang is Lang.RO or lang is Lang.EN or lang is Lang.RU:
        for pt in PronounTypeEnum:
            for text_element in text_element_types:
                for measure_function in measure_functions:
                    indices.append(Pronoun(lang, pt, text_element, measure_function))

    return indices