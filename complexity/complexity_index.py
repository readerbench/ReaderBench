
import abc
from typing import Callable, Iterable, List, Tuple

from rb.cna.cna_graph import CnaGraph
from rb.complexity.index_category import IndexCategory
from rb.complexity.measure_function import (MeasureFunction, average,
                                            standard_deviation, maximum)
from rb.core.document import Document
from rb.core.lang import Lang
from rb.core.text_element import TextElement
from rb.core.text_element_type import TextElementType
from rb.similarity.lda import LDA
from rb.similarity.lsa import LSA
from rb.similarity.vector_model import VectorModel
from rb.similarity.word2vec import Word2Vec
from rb.utils.downloader import download_wordlist
from rb.utils.rblogger import Logger

logger = Logger.get_logger()



"""TODO check for indices which compute on empty set of values """
class ComplexityIndex():
    """General class for any complexity index
    
    Attributes
    ----------
    lang : Lang
        language where the index is applied
    category : IndexCategory
        type of the index e.g. SURFACE, SYNTAX
    abbr : str
        short string describing the index
    reduce_depth : TextElementType
        the depth (in the document) where the reduce_function is applied
        the index is applied recursivley on all the above element types from the document tree
    reduce_function : Callable[[List], float]
        a function to summarize the results of the index (average or standard deviation)

    Methods
    -------
    process(element: TextElement)
        computes the index, overwritten for each index
    
    __repr__()
        detailed string representation of the index, should overwritten by each index
    """

    IDENTITY = 0

    def __init__(self, lang: Lang, category: IndexCategory, abbr: str, reduce_depth: int, reduce_function: MeasureFunction):
        self.lang = lang
        self.category = category
        self.abbr = abbr
        self.reduce_function = reduce_function
        self.reduce_depth = reduce_depth
        if self.reduce_function is None:
            self.reduce_function_abbr =  '' 
        elif self.reduce_function is average:
            self.reduce_function_abbr = 'M'
        elif self.reduce_function is standard_deviation:
            self.reduce_function_abbr =  'SD'
        elif self.reduce_function is maximum:
            self.reduce_function_abbr = 'Max'
        else:
            self.reduce_function_abbr = ''
        self.reduce_depth_abbr = '' if self.reduce_depth is None else self.element_to_abr(
                self.element_type_from_depth(self.reduce_depth).name)

    def element_type_from_depth(self, depth) -> TextElementType:
        for el_type in TextElementType:
            if el_type.value == depth:
                return el_type

    def element_to_abr(self, s) -> str:
        if s == 'BLOCK':
            return 'Par'
        return s[0].upper() + s[1:].lower()

    def process(self, element: TextElement) -> float:
        return self.reduce_function(self._compute(element))

    @abc.abstractmethod
    def _compute_value(self, element: TextElement) -> float:
        pass

    def _compute(self, element: TextElement) -> List[float]:
        if element.depth > self.reduce_depth:
            values = []
            for child in element.components:
                values += self._compute(child)
            element.indices[self] = self.reduce_function(values)
        elif element.depth == self.reduce_depth:
            values = [self._compute_value(element)]
            element.indices[self] = self.reduce_function(values)
        else:
            logger.error('wrong reduce depth value.')
            return []
        return values

    # overwritten by each index 
    def __repr__(self):
        return f"{self.reduce_function_abbr}({self.abbr} / {self.reduce_depth_abbr})"

    def __eq__(self, value):
        return repr(self) == repr(value)

    def __hash__(self):
        return hash(repr(self))
    
    def __getstate__(self):
        return {
            key: value
            for key, value in self.__dict__.items()
            if key != "cna_graph"
        }
        
    def __setstate__(self, state):
        for key, value in state.items():
            setattr(self, key, value)

def compute_index(index: ComplexityIndex, element: TextElement) -> float:
    return index.process(element)

# computed indices and saves for each TextElement in indices dictionary
def compute_indices(doc: Document, cna_graph: CnaGraph = None):
    logger.info('Starting computing all indices for {0} type element'.format(type(doc).__name__))
    download_wordlist(doc.lang)
    for cat in IndexCategory:
        for index in cat.create(doc.lang, cna_graph):
            compute_index(index, doc)
