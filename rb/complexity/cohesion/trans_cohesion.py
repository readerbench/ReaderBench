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
        
    def process(self, element: TextElement) -> float:
        return self.compute(element)

    def compute(self, element: TextElement) -> float:

        blocks = element.get_blocks()
        if len(blocks) < 2:
            element.indices[self] = ComplexityIndex.IDENTITY
            return ComplexityIndex.IDENTITY            
        else:
            sim_values = []            
            for i, _ in enumerate(blocks[:-1]):
                current_block_sents = blocks[i].get_sentences()
                next_block_sents = blocks[i+1].get_sentences()
                if len(current_block_sents) == 0 or len(next_block_sents) == 0:
                    continue
                cur_sent = current_block_sents[-1]
                next_sent = next_block_sents[0]
                sim_edge = self.cna_graph.edges(node=(cur_sent, next_sent), edge_type=EdgeType.SEMANTIC,
                                                vector_model=None)
                v = sim_edge[0][2]
                sim_values.append(v)
            element.indices[self] = sum(sim_values) / len(sim_values) if len(sim_values) > 1 else ComplexityIndex.IDENTITY
            return element.indices[self]