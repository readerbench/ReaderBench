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
    from rb.complexity.discourse.connectors import Connector
    from rb.complexity.discourse.conn_type_enum import ConnTypeEnum
    
    indices = []

    if lang is Lang.RO or lang is Lang.EN:
        for ct in ConnTypeEnum:
            indices.append(Connector(lang, ct, TextElementType.SENT.value, MeasureFunction.AVG))
            indices.append(Connector(lang, ct, TextElementType.BLOCK.value, MeasureFunction.AVG))
    return indices