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
        
    def process(self, element: TextElement) -> float:
        return self.compute(element)

    def compute(self, element: TextElement) -> float:

        blocks = element.get_blocks()
        if len(blocks) < 3:
            element.indices[self] = ComplexityIndex.IDENTITY
            return ComplexityIndex.IDENTITY            
        else:
            start_block = blocks[0]
            scale_factor, weighted_sum = 0, 0
            for i, _ in enumerate(blocks):
                if i == 0:  continue
                sim_edge = self.cna_graph.edges(node=(start_block, blocks[i]), edge_type=EdgeType.SEMANTIC, 
                                                vector_model=None)
                v = sim_edge[0][2]
                weighted_sum += v * (1.0 / i)
                scale_factor += (1.0 / i)
            element.indices[self] = weighted_sum / scale_factor 
            return element.indices[self]