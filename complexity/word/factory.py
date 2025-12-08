from typing import List

from rb.core.lang import Lang
from rb.complexity.measure_function import MeasureFunction
from rb.core.text_element_type import TextElementType
from rb.complexity.measure_function import MeasureFunction
from rb.cna.cna_graph import CnaGraph

# create all indices
# dependencies need to be putted in function because otherwise circular dependencies happens
def create(lang: Lang, cna_graph: CnaGraph) -> List["ComplexityIndex"]:
    from rb.complexity.word.wd_len import WdLen
    from rb.complexity.word.wd_diff_lemma import WdDiffLemma
    from rb.complexity.word.no_repetitions import NoRepetitions
    from rb.complexity.word.name_entity_enum import NamedEntityONEnum
    from rb.complexity.word.name_entity import NoNamedEntity
    from rb.complexity.word.wd_max_depth_hypernym_tree import WdMaxDpthHypTree
    from rb.complexity.word.wd_avg_depth_hypernym_tree import WdAvgDpthHypTree
    from rb.complexity.word.no_wd_paths_hypernym_tree import NoWdPathsHypTree
    from rb.complexity.word.polysemy import Polysemy
    from rb.complexity.word.wd_syllab import WdSyllab
    from rb.complexity.word.aoa import Aoa
    from rb.complexity.word.aoa_enum import AoaTypeEnum
    from rb.complexity.word.aoe import Aoe
    from rb.complexity.word.aoe_enum import AoeTypeEnum
    from rb.complexity.word.valence import Valence
    from rb.complexity.word.valence_type import ValenceTypeEnum

    indices = []
    
    text_elements = [TextElementType.WORD.value, TextElementType.SENT.value, TextElementType.BLOCK.value]
    measure_functions = [MeasureFunction.AVG, MeasureFunction.STDEV, MeasureFunction.MAX]
    word_indices = [WdLen, WdSyllab, WdDiffLemma, WdMaxDpthHypTree, WdAvgDpthHypTree, NoWdPathsHypTree, Polysemy]
    
    for index_class in word_indices:
        for measure_function in measure_functions:
            indices.append(index_class(lang, TextElementType.WORD.value, measure_function))

    if lang is Lang.EN:
        for at in AoaTypeEnum:
            for text_element in text_elements:
                for measure_function in measure_functions:
                    indices.append(Aoa(lang, at, text_element, measure_function))
            indices.append(Aoa(lang, at, TextElementType.DOC.value, MeasureFunction.IDENTITY))
        for at in AoeTypeEnum:
            for text_element in text_elements:
                for measure_function in measure_functions:
                    indices.append(Aoe(lang, at, text_element, measure_function))
            indices.append(Aoe(lang, at, TextElementType.DOC.value, MeasureFunction.IDENTITY))
        for vt in ValenceTypeEnum:
            for text_element in text_elements:
                for measure_function in measure_functions:
                    indices.append(Valence(lang, vt, text_element, measure_function))
            indices.append(Valence(lang, vt, TextElementType.DOC.value, MeasureFunction.IDENTITY))

    for named_ent_type in NamedEntityONEnum:
        for text_element in text_elements[1:]:
            for measure_function in measure_functions:
                indices.append(NoNamedEntity(lang, named_ent_type=named_ent_type, reduce_depth=text_element, 
                                    reduce_function=measure_function))
        indices.append(NoNamedEntity(lang, named_ent_type=named_ent_type, reduce_depth=TextElementType.DOC.value, 
                                        reduce_function=MeasureFunction.IDENTITY))
    for text_element in text_elements[1:]:
        for measure_function in measure_functions:
            indices.append(NoRepetitions(lang, reduce_depth=text_element, reduce_function=measure_function))
        indices.append(NoRepetitions(lang, reduce_depth=TextElementType.DOC.value, reduce_function=MeasureFunction.IDENTITY))
    return indices