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

    indices.append(WdLen(lang, TextElementType.WORD.value, MeasureFunction.AVG))
    indices.append(WdLen(lang, TextElementType.SENT.value, MeasureFunction.STDEV))

    indices.append(WdSyllab(lang, reduce_depth=TextElementType.SENT.value, reduce_function=MeasureFunction.AVG))
    indices.append(WdSyllab(lang, reduce_depth=TextElementType.SENT.value, reduce_function=MeasureFunction.STDEV))

    indices.append(WdSyllab(lang, reduce_depth=TextElementType.BLOCK.value, reduce_function=MeasureFunction.AVG))
    indices.append(WdSyllab(lang, reduce_depth=TextElementType.BLOCK.value, reduce_function=MeasureFunction.STDEV))

    indices.append(WdDiffLemma(lang, reduce_depth=TextElementType.WORD.value, reduce_function=MeasureFunction.AVG))
    
    indices.append(NoRepetitions(lang, window_size=8, reduce_depth=TextElementType.SENT.value, 
                                    reduce_function=MeasureFunction.AVG))
    indices.append(WdMaxDpthHypTree(lang, reduce_depth=TextElementType.WORD.value, 
                                    reduce_function=MeasureFunction.AVG))
    indices.append(WdAvgDpthHypTree(lang, reduce_depth=TextElementType.WORD.value, 
                                    reduce_function=MeasureFunction.AVG))
    indices.append(NoWdPathsHypTree(lang, reduce_depth=TextElementType.WORD.value, 
                                    reduce_function=MeasureFunction.AVG))
    indices.append(Polysemy(lang, reduce_depth=TextElementType.WORD.value, 
                                    reduce_function=MeasureFunction.AVG))

    if lang is Lang.EN:
        for at in AoaTypeEnum:
            indices.append(Aoa(lang, at, reduce_depth=TextElementType.WORD.value, 
                                    reduce_function=MeasureFunction.AVG))
            indices.append(Aoa(lang, at, reduce_depth=TextElementType.WORD.value, 
                                    reduce_function=MeasureFunction.STDEV))
        for at in AoeTypeEnum:
            indices.append(Aoe(lang, at, reduce_depth=TextElementType.WORD.value, 
                                    reduce_function=MeasureFunction.AVG))
            indices.append(Aoe(lang, at, reduce_depth=TextElementType.WORD.value, 
                                    reduce_function=MeasureFunction.STDEV))
        for vt in ValenceTypeEnum:
            indices.append(Valence(lang, vt, reduce_depth=TextElementType.SENT.value, 
                                    reduce_function=MeasureFunction.AVG))
            indices.append(Valence(lang, vt, reduce_depth=TextElementType.BLOCK.value, 
                                    reduce_function=MeasureFunction.AVG))

    for named_ent_type in NamedEntityONEnum:
        indices.append(NoNamedEntity(lang, named_ent_type=named_ent_type, reduce_depth=TextElementType.WORD.value, 
                                        reduce_function=MeasureFunction.AVG))
    
    return indices