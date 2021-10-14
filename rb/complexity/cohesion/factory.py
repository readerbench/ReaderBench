from typing import List

from typing import List

from rb.core.lang import Lang
from rb.complexity.measure_function import MeasureFunction
from rb.core.text_element_type import TextElementType
from rb.complexity.measure_function import MeasureFunction
from rb.cna.cna_graph import CnaGraph

# create all indices
# dependencies need to be putted in function because otherwise circular dependencies happens
def create(lang: Lang, cna_graph: CnaGraph) -> List["ComplexityIndex"]:
    from rb.complexity.cohesion.adj_cohesion import AdjCohesion
    # from rb.complexity.cohesion.adj_ext_cohesion import AdjExternalCohesion
    from rb.complexity.cohesion.intra_cohesion import IntraCohesion
    from rb.complexity.cohesion.start_end_cohesion import StartEndCohesion
    from rb.complexity.cohesion.start_mid_cohesion import StartMiddleCohesion
    from rb.complexity.cohesion.mid_end_cohesion import MiddleEndCohesion
    from rb.complexity.cohesion.trans_cohesion import TransCohesion
    from rb.complexity.cohesion.inter_cohesion import InterCohesion

    indices = []
    if cna_graph:
        indices.append(AdjCohesion(lang=lang, element_type=TextElementType.SENT, 
                                        reduce_depth=TextElementType.DOC.value, reduce_function=MeasureFunction.IDENTITY,
                                        cna_graph=cna_graph))
        indices.append(AdjCohesion(lang=lang, element_type=TextElementType.BLOCK, 
                                        reduce_depth=TextElementType.DOC.value, reduce_function=MeasureFunction.IDENTITY,
                                        cna_graph=cna_graph))
        indices.append(AdjCohesion(lang=lang, element_type=TextElementType.SENT, 
                                        reduce_depth=TextElementType.BLOCK.value, reduce_function=MeasureFunction.AVG,
                                        cna_graph=cna_graph))
        indices.append(AdjCohesion(lang=lang, element_type=TextElementType.SENT, 
                                        reduce_depth=TextElementType.BLOCK.value, reduce_function=MeasureFunction.STDEV,
                                        cna_graph=cna_graph))
        indices.append(AdjCohesion(lang=lang, element_type=TextElementType.SENT, 
                                        reduce_depth=TextElementType.BLOCK.value, reduce_function=MeasureFunction.MAX,
                                        cna_graph=cna_graph))
        if cna_graph.pairwise:
            indices.append(IntraCohesion(lang=lang, element_type=TextElementType.BLOCK, 
                                            reduce_depth=TextElementType.BLOCK.value, reduce_function=MeasureFunction.AVG,
                                            cna_graph=cna_graph))
            indices.append(IntraCohesion(lang=lang, element_type=TextElementType.BLOCK, 
                                            reduce_depth=TextElementType.BLOCK.value, reduce_function=MeasureFunction.STDEV,
                                            cna_graph=cna_graph))
            indices.append(IntraCohesion(lang=lang, element_type=TextElementType.BLOCK, 
                                            reduce_depth=TextElementType.BLOCK.value, reduce_function=MeasureFunction.MAX,
                                            cna_graph=cna_graph))
            indices.append(InterCohesion(lang=lang, element_type=TextElementType.DOC, 
                                            reduce_depth=TextElementType.DOC.value, reduce_function=MeasureFunction.IDENTITY,
                                            cna_graph=cna_graph))
            indices.append(StartEndCohesion(lang=lang,
                                            reduce_depth=TextElementType.DOC.value, reduce_function=MeasureFunction.IDENTITY,
                                            cna_graph=cna_graph))
            indices.append(StartEndCohesion(lang=lang,
                                            reduce_depth=TextElementType.BLOCK.value, reduce_function=MeasureFunction.AVG,
                                            cna_graph=cna_graph))
            indices.append(StartEndCohesion(lang=lang,
                                            reduce_depth=TextElementType.BLOCK.value, reduce_function=MeasureFunction.STDEV,
                                            cna_graph=cna_graph))
            indices.append(StartEndCohesion(lang=lang,
                                            reduce_depth=TextElementType.BLOCK.value, reduce_function=MeasureFunction.MAX,
                                            cna_graph=cna_graph))
            indices.append(StartMiddleCohesion(lang=lang,
                                            reduce_depth=TextElementType.DOC.value, reduce_function=MeasureFunction.IDENTITY,
                                            cna_graph=cna_graph))
            indices.append(StartMiddleCohesion(lang=lang,
                                            reduce_depth=TextElementType.BLOCK.value, reduce_function=MeasureFunction.AVG,
                                            cna_graph=cna_graph))
            indices.append(StartMiddleCohesion(lang=lang,
                                            reduce_depth=TextElementType.BLOCK.value, reduce_function=MeasureFunction.STDEV,
                                            cna_graph=cna_graph))
            indices.append(StartMiddleCohesion(lang=lang,
                                            reduce_depth=TextElementType.BLOCK.value, reduce_function=MeasureFunction.MAX,
                                            cna_graph=cna_graph))
            indices.append(MiddleEndCohesion(lang=lang,
                                            reduce_depth=TextElementType.DOC.value, reduce_function=MeasureFunction.IDENTITY,
                                            cna_graph=cna_graph))
            indices.append(MiddleEndCohesion(lang=lang,
                                            reduce_depth=TextElementType.BLOCK.value, reduce_function=MeasureFunction.AVG,
                                            cna_graph=cna_graph))
            indices.append(MiddleEndCohesion(lang=lang,
                                            reduce_depth=TextElementType.BLOCK.value, reduce_function=MeasureFunction.STDEV,
                                            cna_graph=cna_graph))
            indices.append(MiddleEndCohesion(lang=lang,
                                            reduce_depth=TextElementType.BLOCK.value, reduce_function=MeasureFunction.MAX,
                                            cna_graph=cna_graph))
            indices.append(TransCohesion(lang=lang,
                                            reduce_depth=TextElementType.BLOCK.value, reduce_function=MeasureFunction.AVG,
                                            cna_graph=cna_graph))
            indices.append(TransCohesion(lang=lang,
                                            reduce_depth=TextElementType.BLOCK.value, reduce_function=MeasureFunction.STDEV,
                                            cna_graph=cna_graph))
            indices.append(TransCohesion(lang=lang,
                                            reduce_depth=TextElementType.BLOCK.value, reduce_function=MeasureFunction.MAX,
                                            cna_graph=cna_graph))
                                
    return indices