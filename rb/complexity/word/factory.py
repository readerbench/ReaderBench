from typing import List

from rb.core.lang import Lang
from rb.complexity.measure_function import MeasureFunction
from rb.core.text_element_type import TextElementType
from rb.complexity.measure_function import MeasureFunction
# create all indices
# dependencies need to be putted in function because otherwise circular dependencies happens
def create(lang: Lang) -> List["ComplexityIndex"]:
    from rb.complexity.word.wd_len import WdLen
    from rb.complexity.word.wd_diff_lemma import WdDiffLemma
    from rb.complexity.word.no_repetitions import NoRepetitions
    from rb.complexity.word.name_entity_enum import NamedEntityONEnum
    from rb.complexity.word.name_entity import NoNamedEntity
    from rb.complexity.word.wd_max_depth_hypernym_tree import WdMaxDpthHypTree
    from rb.complexity.word.wd_avg_depth_hypernym_tree import WdAvgDpthHypTree
    from rb.complexity.word.no_wd_paths_hypernym_tree import NoWdPathsHypTree
    from rb.complexity.word.polysemy import Polysemy
    
    indices = []
    indices.append(WdLen(lang, TextElementType.WORD.value, MeasureFunction.AVG))
    indices.append(WdDiffLemma(lang, TextElementType.WORD.value, MeasureFunction.AVG))
    indices.append(NoRepetitions(lang, window_size=8, reduce_depth=TextElementType.SENT.value, 
                                    reduce_function=MeasureFunction.AVG))
    # indices.append(NoNamedEntity(lang, named_ent_type=NamedEntityONEnum.NORP, reduce_depth=TextElementType.WORD.value, 
    #                                 reduce_function=MeasureFunction.AVG))
    indices.append(WdMaxDpthHypTree(lang, reduce_depth=TextElementType.WORD.value, 
                                    reduce_function=MeasureFunction.AVG))
    indices.append(WdAvgDpthHypTree(lang, reduce_depth=TextElementType.WORD.value, 
                                    reduce_function=MeasureFunction.AVG))
    indices.append(NoWdPathsHypTree(lang, reduce_depth=TextElementType.WORD.value, 
                                    reduce_function=MeasureFunction.AVG))
    indices.append(Polysemy(lang, reduce_depth=TextElementType.WORD.value, 
                                    reduce_function=MeasureFunction.AVG))
    #indices = [DepIndex(lang, dep, TextElementType.BLOCK, MeasureFunction.AVG) for dep in DepEnum]
    #indices += [DepIndex(lang, dep, TextElementType.BLOCK, MeasureFunction.STDEV) for dep in DepEnum]
    return indices