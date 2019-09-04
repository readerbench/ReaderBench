from typing import List

from typing import List

from rb.core.lang import Lang
from rb.complexity.measure_function import MeasureFunction
from rb.core.text_element_type import TextElementType
from rb.complexity.measure_function import MeasureFunction

# create all indices
# dependencies need to be putted in function because otherwise circular dependencies happens
def create(lang: Lang) -> List["ComplexityIndex"]:
    from rb.complexity.cohesion.adj_cohesion import AdjCohesion
    from rb.complexity.cohesion.intra_cohesion import IntraCohesion
    from rb.complexity.cohesion.start_end_cohesion import StartEndCohesion

    indices = []
    indices.append(IntraCohesion(lang=lang, element_type=TextElementType.BLOCK, 
                                reduce_depth=TextElementType.BLOCK.value, reduce_function=MeasureFunction.AVG))
    """ inter cohesion """
    indices.append(IntraCohesion(lang=lang, element_type=TextElementType.DOC, 
                                reduce_depth=TextElementType.DOC.value, reduce_function=MeasureFunction.AVG))
    indices.append(AdjCohesion(lang=lang, element_type=TextElementType.DOC, 
                                reduce_depth=TextElementType.DOC.value, reduce_function=MeasureFunction.AVG))
    indices.append(AdjCohesion(lang=lang, element_type=TextElementType.BLOCK, 
                                reduce_depth=TextElementType.DOC.value, reduce_function=MeasureFunction.AVG))
    indices.append(StartEndCohesion(lang=lang,
                                reduce_depth=None, reduce_function=None))
    return indices