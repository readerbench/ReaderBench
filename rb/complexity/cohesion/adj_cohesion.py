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


class AdjCohesion(ComplexityIndex):

    """AdjCohesion between text elements of element_type """
    def __init__(self, lang: Lang, element_type: TextElementType,
            reduce_depth: int, reduce_function: MeasureFunction, cna_graph: CnaGraph):
        ComplexityIndex.__init__(self, lang=lang, category=IndexCategory.COHESION,
                                 reduce_depth=reduce_depth, reduce_function=reduce_function,
                                 abbr="AdjCoh")
        self.element_type = element_type
        self.cna_graph = cna_graph
        if element_type.value > reduce_depth:
            logger.error('For index {} element_type has to be lower or equal than reduce_depth'.format(self))

    def _compute_value(self, element: TextElement) -> float:
        sim_values = []
        if self.element_type is TextElementType.SENT:
            children = element.get_sentences()
        else:
            children = element.components
        if len(children) < 2:
            return 0
        for i, child in enumerate(children[:-1]): 
            _, _, sim = self.cna_graph.edges(node=(child, children[i+1]), edge_type=EdgeType.SEMANTIC, vector_model=None)[0]
            sim_values.append(sim)
        return mean(sim_values)
    
    def __repr__(self):
        return f"{self.reduce_function_abbr}({self.element_to_abr(self.element_type.name)}{self.abbr} / {self.reduce_depth_abbr})"