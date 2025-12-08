from rb.complexity.complexity_index import ComplexityIndex
from rb.complexity.index_category import IndexCategory
from rb.complexity.measure_function import MeasureFunction
from rb.complexity.rhythm.syllabified_dict import SyllabifiedDict
from rb.core.lang import Lang
from rb.core.text_element import TextElement
from rb.utils.rblogger import Logger

from typing import List
from functools import reduce

logger = Logger.get_logger()


class NoStressedSyllablesIndex(ComplexityIndex):

    def __init__(self, lang: Lang, reduce_depth: int,
                 reduce_function: MeasureFunction):

        ComplexityIndex.__init__(self, lang=lang, category=IndexCategory.RHYTHM,
                                 abbr="RhNoStressedSyll", reduce_depth=reduce_depth,
                                 reduce_function=reduce_function)

        self.syllabified_dict = SyllabifiedDict.get_instance(lang)

    def process(self, element: TextElement) -> float:
        return self.reduce_function(self.compute_above(element))

    def compute_below(self, element: TextElement) -> float:
        if element.is_word():
            if element.text in self.syllabified_dict:
                # a word can be syllabified in multiple ways
                syllabified_words = self.syllabified_dict[element.text]
                if len(syllabified_words) == 1:
                    syllables = syllabified_words[0]
                    return reduce(lambda syllable: 1 if any(phoneme.endswith('1') for phoneme in syllable) else 0,
                                  syllables)
                else:
                    # TODO: choose the correct version
                    pass
            else:
                return 1
        elif element.depth <= self.reduce_depth:
            res = 0
            for child in element.components:
                res += self.compute_below(child)
            return res

    def compute_above(self, element: TextElement) -> List[float]:
        values = list()
        if element.depth > self.reduce_depth:
            for child in element.components:
                values += self.compute_above(child)
            element.indices[self] = self.reduce_function(values)
        elif element.depth == self.reduce_depth:
            values = [self.compute_below(element)]
        else:
            logger.error("wrong reduce depth value.")
        return values
