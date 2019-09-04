from rb.complexity.complexity_index import ComplexityIndex
from rb.core.lang import Lang
from rb.core.text_element import TextElement
from rb.complexity.index_category import IndexCategory
from rb.complexity.measure_function import MeasureFunction
from rb.core.text_element_type import TextElementType   
from typing import List, Callable
from rb.similarity.vector_model import VectorModel

from rb.utils.rblogger import Logger

logger = Logger.get_logger()


class StartEndCohesion(ComplexityIndex):

    """ only between start block and end block """
    def __init__(self, lang: Lang,
            reduce_depth: int = None, reduce_function: MeasureFunction = None):
        ComplexityIndex.__init__(self, lang=lang, category=IndexCategory.COHESION,
                                 reduce_depth=reduce_depth, reduce_function=reduce_function,
                                 abbr="StartEndCoh")

    def process(self, element: TextElement) -> float:
        return self.compute(element)

    def compute(self, element: TextElement) -> float:

        doc = element.get_parent_document()
        if len(doc.get_blocks()) < 2:
            element.indices[self] = ComplexityIndex.IDENTITY
            values = []
        else:
            start_block = element.get_blocks()[0]
            end_block = element.get_blocks()[-1]
            v = doc.cna_graph.model.similarity(start_block, end_block)
            element.indices[self] = v
            return v