from rb.complexity.complexity_index import ComplexityIndex
from rb.core.lang import Lang
from rb.core.text_element import TextElement
from rb.complexity.index_category import IndexCategory
from rb.complexity.measure_function import MeasureFunction
from rb.core.text_element_type import TextElementType   
from rb.core.sentence import Sentence
from typing import List, Callable
from rb.utils.rblogger import Logger

logger = Logger.get_logger()


class NoRepetitions(ComplexityIndex):

    
    def __init__(self, lang: Lang, reduce_depth: int, 
                 reduce_function: MeasureFunction, window_size: int = 8):

        ComplexityIndex.__init__(self, lang=lang, category=IndexCategory.WORD,
                                 abbr="Repetitions", reduce_depth=reduce_depth,
                                 reduce_function=reduce_function)
        self.window_size = window_size

    def compute_repetitions(self, sent: Sentence):
        # TODO count also synonyms, not just same lemmas
        count_reps = 0

        for start in range(max(1, len(sent.components) - self.window_size)):
            word1 = sent.components[start]
            for i in range(min(len(sent.components) - 1, start + 1), min(len(sent.components), start + self.window_size)):
                word2 = sent.components[i]
                if word1.lemma == word2.lemma:
                    count_reps += 1

        return count_reps

    def _compute_value(self, element: TextElement) -> int:
        return sum(self.compute_repetitions(sent) for sent in element.get_sentences())