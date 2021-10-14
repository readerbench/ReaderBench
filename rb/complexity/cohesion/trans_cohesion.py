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


class TransCohesion(ComplexityIndex):

    def __init__(self, lang: Lang,
            reduce_depth: int = None, reduce_function: MeasureFunction = None,
            cna_graph: CnaGraph = None):
        ComplexityIndex.__init__(self, lang=lang, category=IndexCategory.COHESION,
                                 reduce_depth=reduce_depth, reduce_function=reduce_function,
                                 abbr="TransCoh")
        self.cna_graph = cna_graph
        
    def _compute_value(self, element: TextElement) -> float:
        i = element.index_in_container
        if i == len(element.container.components) - 1:
            return None
        if len(element.components) < 2:
            return None
        current_block_sents = element.get_sentences()
        next_block_sents = element.container.components[i+1].get_sentences()
        if len(current_block_sents) == 0 or len(next_block_sents) == 0:
            return None
        cur_sent = current_block_sents[-1]
        next_sent = next_block_sents[0]
        sim_edge = self.cna_graph.edges(node=(cur_sent, next_sent), edge_type=EdgeType.SEMANTIC,
                                        vector_model=None)
        if sim_edge:
            return sim_edge[0][2]
        else:
            return None