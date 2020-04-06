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

    
    def __init__(self, lang: Lang, window_size: int, 
        reduce_depth: int, reduce_function: MeasureFunction):

        ComplexityIndex.__init__(self, lang=lang, category=IndexCategory.WORD,
                                 abbr="Repetitions", reduce_depth=reduce_depth,
                                 reduce_function=reduce_function)
        self.window_size = window_size

    def process(self, element: TextElement) -> float:
        return self.reduce_function(self.compute_above(element))

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


    def compute_below(self, element: TextElement) -> List[float]:
        if element.is_sentence() == True:
            res = self.compute_repetitions(element)
            return [res]
        elif element.depth <= self.reduce_depth:
            res = []
            for child in element.components:
                res += self.compute_below(child)
            return res
    
    def compute_above(self, element: TextElement) -> List[float]:
        if element.depth > self.reduce_depth:
            values = []
            for child in element.components:
                values += self.compute_above(child)
            element.indices[self] = self.reduce_function(values)
        elif element.depth == self.reduce_depth:
            values = self.compute_below(element)
            element.indices[self] = self.reduce_function(values)
        else:
            logger.error('wrong reduce depth value.')
        return values