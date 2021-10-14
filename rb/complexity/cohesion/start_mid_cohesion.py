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


class StartMiddleCohesion(ComplexityIndex):

    """ only between start block and middle block """
    def __init__(self, lang: Lang,
            reduce_depth: int = None, reduce_function: MeasureFunction = None,
            cna_graph: CnaGraph = None):
        ComplexityIndex.__init__(self, lang=lang, category=IndexCategory.COHESION,
                                 reduce_depth=reduce_depth, reduce_function=reduce_function,
                                 abbr="StartMidCoh")
        self.cna_graph = cna_graph
        
    def _compute_value(self, element: TextElement) -> float:
        if len(element.components) < 3:
            return None
        start = element.components[0]
        scale_factor, weighted_sum = 0, 0

        for i, elem in enumerate(element.components[1:]):
            sim_edge = self.cna_graph.edges(node=(start, elem), edge_type=EdgeType.SEMANTIC,
                                            vector_model=None)
            if sim_edge:
                v = sim_edge[0][2]
                weighted_sum += v * (1.0 / (i + 1))
                scale_factor += (1.0 / (i + 1))
        return weighted_sum / scale_factor