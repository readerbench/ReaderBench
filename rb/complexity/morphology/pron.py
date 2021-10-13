from rb.complexity.complexity_index import ComplexityIndex
from rb.core.lang import Lang
from rb.core.text_element import TextElement
from rb.complexity.index_category import IndexCategory
from rb.complexity.measure_function import MeasureFunction
from rb.core.text_element_type import TextElementType
from rb.complexity.morphology.pron_type_enum import PronounTypeEnum
from typing import List, Callable, Set
from rb.core.pos import POS as PosEum
from rb.utils.rblogger import Logger
from typing import Dict, List
logger = Logger.get_logger()


class Pronoun(ComplexityIndex):
    
    
    pronoun_dict: Dict[PronounTypeEnum, List[str]] = None
    
    def __init__(self, lang: Lang, pron_type: PronounTypeEnum,
            reduce_depth: int, reduce_function: MeasureFunction):

        ComplexityIndex.__init__(self, lang=lang, category=IndexCategory.MORPHOLOGY,
                                 abbr="Pron", reduce_depth=reduce_depth,
                                 reduce_function=reduce_function)
        if Pronoun.pronoun_dict is None:
            Pronoun.parse_pronoun_list(lang)

        self.pron_type = pron_type


    @staticmethod
    def parse_pronoun_list(lang: Lang):

        if lang is Lang.RO:
            path_to_wordlist = 'resources/ro/wordlists/pronouns_ro.txt'
        elif lang is Lang.EN:
            path_to_wordlist = 'resources/en/wordlists/pronouns_en.txt'
        elif lang is Lang.RU:
            path_to_wordlist = 'resources/ru/wordlists/pronouns_ru.txt'

        with open(path_to_wordlist, 'rt', encoding='utf-8') as f:
            Pronoun.pronoun_dict = {}
            for pt in PronounTypeEnum:
                Pronoun.pronoun_dict[pt] = []

            pron_type = None
            for i, line in enumerate(f):
                if line.find('first_person') != -1:
                    pron_type = PronounTypeEnum.FST
                elif line.find('second_person') != -1:
                    pron_type = PronounTypeEnum.SND
                elif line.find('third_person') != -1:
                    pron_type = PronounTypeEnum.THRD
                elif line.find('interrogative') != -1:
                    pron_type = PronounTypeEnum.INT
                elif line.find('indefinite') != -1:
                    pron_type = PronounTypeEnum.INDEF
                elif len(line.strip()) == 0:
                    continue
                elif pron_type is not None:
                    Pronoun.pronoun_dict[pron_type].append(line.strip())

    def process(self, element: TextElement) -> float:
        return self.reduce_function(self.compute_above(element))

    def compute_below(self, element: TextElement) -> List[str]:
        if element.is_word() == True:
            res = []
            if element.text.lower() in Pronoun.pronoun_dict[self.pron_type]:
                res.append(element.text)
            return res
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
            element.indices[self] = len(self.compute_below(element))
            values = [len(self.compute_below(element))]
            element.indices[self] = self.reduce_function(values)
        else:
            logger.error('wrong reduce depth value.')
        return values

    def __repr__(self):
        return f"{self.reduce_function_abbr}({self.abbr}_{self.pron_type.name.lower()} / {self.reduce_depth_abbr})"
