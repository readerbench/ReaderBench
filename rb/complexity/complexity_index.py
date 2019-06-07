
from multiprocessing import Pool, cpu_count
from typing import Callable, Iterable, List, Tuple

from rb.complexity.index_category import IndexCategory
from rb.complexity.measure_function import MeasureFunction
from rb.core.lang import Lang
from rb.core.text_element import TextElement
from rb.core.text_element_type import TextElementType
from rb.utils.rblogger import Logger
from rb.similarity.lda import LDA
from rb.similarity.lsa import LSA
from rb.similarity.word2vec import Word2Vec
from joblib import Parallel, delayed

logger = Logger.get_logger()

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

    def __init__(self, lang: Lang, category: IndexCategory, abbr: str, reduce_depth: int, reduce_function: MeasureFunction):
        self.lang = lang
        self.category = category
        self.abbr = abbr
        self.reduce_function = reduce_function
        self.reduce_depth = reduce_depth

    # overwritten by each index
    def process(self, element: TextElement) -> float:
        pass

    # overwritten by each index 
    def __repr__(self):
        return self.abbr

def compute_index(index: ComplexityIndex, element: TextElement) -> float:
    return index.process(element)

# computed indices and saves for each TextElement in indices dictionary
def compute_indices(element: TextElement):
    logger.info('Starting computing all indices for {0} type element'.format(type(element).__name__))
    num_cores = cpu_count()
    # for cat in IndexCategory:
    #     for index in cat.value(element.lang):
    #         index.process(element)
    # with Pool(processes=num_cores) as pool:
    # tasks = [(index, element) for cat in IndexCategory for index in cat.create(element.lang)]
    lda = LDA('coca', Lang.EN)
    Parallel(n_jobs=num_cores, prefer="threads")(delayed(compute_index)(index, element) for cat in IndexCategory for index in cat.create(element.lang))
        