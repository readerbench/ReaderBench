""" Cohesion between current sentence/block and its neighbours. This is different from adj_cojesion
which averages/stdev cohesions on block or doc level. There is no avgstddev here.  """

from rb.complexity.complexity_index import ComplexityIndex
from rb.core.lang import Lang
from rb.core.text_element import TextElement
from rb.complexity.index_category import IndexCategory
from rb.complexity.measure_function import MeasureFunction
from rb.core.text_element_type import TextElementType   
from typing import List, Callable
from rb.similarity.vector_model import VectorModel
from rb.cna.cna_graph import CnaGraph

from rb.utils.rblogger import Logger

logger = Logger.get_logger()

class AdjExternalCohesion(ComplexityIndex):

    def __init__(self, lang: Lang, element_type: TextElementType,
            reduce_depth: int = None, reduce_function: MeasureFunction = None,
            cna_graph: CnaGraph = None):
        ComplexityIndex.__init__(self, lang=lang, category=IndexCategory.COHESION,
                                 reduce_depth=reduce_depth, reduce_function=reduce_function,
                                 abbr="AdjExtCoh")
        self.element_type = element_type        
        if not (element_type is TextElementType.BLOCK or element_type is TextElementType.SENT):
            logger.error('For index {} element_type has to {} or {}'.format(TextElementType.BLOCK, TextElementType.SENT))

    def process(self, element: TextElement) -> float:
        
        if self.element_type is TextElementType.SENT:
            elems = element.get_sentences()
        else:
            elems = element.get_blocks()

        if len(elems) == 0:
            return ComplexityIndex.IDENTITY
        elif len(elems) == 1:
            elems[0].indices[self] = ComplexityIndex.IDENTITY
            return ComplexityIndex.IDENTITY
        elif len(elems) == 2:
            v = element.get_parent_document().cna_graph.model.similarity(
                                elems[0], elems[1])
            elems[0].indices[self] = elems[1].indices[self] = v
            return ComplexityIndex.IDENTITY
        else:
            for i, elem in enumerate(elems):
                if i == 0:
                    v = element.get_parent_document().cna_graph.model.similarity(elems[i], elems[i + 1])
                    elems[i].indices[self] = v
                elif i == len(elems) - 1:
                    v = element.get_parent_document().cna_graph.model.similarity(elems[i], elems[i - 1])
                    elems[i].indices[self] = v
                else:
                    v = element.get_parent_document().cna_graph.model.similarity(elems[i], elems[i - 1])
                    v += element.get_parent_document().cna_graph.model.similarity(elems[i], elems[i + 1])
                    v /= 2
                    elems[i].indices[self] = v
            return ComplexityIndex.IDENTITY     
    
    def __repr__(self):
        return self.abbr + '_' + self.element_type.name