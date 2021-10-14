from statistics import mean
from rb.complexity.complexity_index import ComplexityIndex
from rb.core.lang import Lang
from rb.core.text_element import TextElement
from rb.complexity.index_category import IndexCategory
from rb.complexity.measure_function import MeasureFunction
from rb.core.text_element_type import TextElementType   
from typing import List, Callable
from rb.similarity.vector_model import VectorModel
from rb.cna.cna_graph import CnaGraph
from rb.cna.edge_type import EdgeType

from rb.utils.rblogger import Logger

logger = Logger.get_logger()


class InterCohesion(ComplexityIndex):

    """InterCohesion between elements element_type"""
    def __init__(self, lang: Lang, element_type: TextElementType,
            reduce_depth: int, reduce_function: MeasureFunction, cna_graph: CnaGraph):
        ComplexityIndex.__init__(self, lang=lang, category=IndexCategory.COHESION,
                                 reduce_depth=reduce_depth, reduce_function=reduce_function,
                                 abbr="InterParCoh")
        self.element_type = element_type
        self.cna_graph = cna_graph    
        if element_type.value > reduce_depth:
            logger.error('For index {} element_type has to be lower or equal than reduce_depth'.format(self))

    def _compute_value(self, element: TextElement) -> float:
        if len(element.components) < 2:
            return True
        sim_values = []
        for i, a in enumerate(element.components[:-1]):
            for b in element.components[(i+1):]:
                sim_edge = self.cna_graph.edges(node=(a, b), edge_type=EdgeType.SEMANTIC, vector_model=None)
                if len(sim_edge) > 0 and len(sim_edge[0]) >= 3:
                    sim_values.append(sim_edge[0][2])
        return mean(sim_values)
    
    def __repr__(self):
        return f"{self.reduce_function_abbr}({self.abbr} / {self.reduce_depth_abbr})"