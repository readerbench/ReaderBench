from typing import List, Callable, Dict
import csv
from rb.complexity.complexity_index import ComplexityIndex
from rb.core.lang import Lang
from rb.core.text_element import TextElement
from rb.complexity.index_category import IndexCategory
from rb.complexity.measure_function import MeasureFunction
from rb.core.text_element_type import TextElementType
from rb.complexity.word.valence_type import ValenceTypeEnum
from rb.utils.rblogger import Logger

logger = Logger.get_logger()


class Valence(ComplexityIndex):


    valence_dict: Dict[str, Dict[str, float]] = None

    def __init__(self, lang: Lang, valence_type: ValenceTypeEnum,
                 reduce_depth: int, reduce_function: MeasureFunction):

        ComplexityIndex.__init__(self, lang=lang, category=IndexCategory.WORD,
                                 abbr="Valence", reduce_depth=reduce_depth,
                                 reduce_function=reduce_function)
        if Valence.valence_dict is None:
            Valence.parse_valence_csv(lang)
        self.valence_type = valence_type

    @staticmethod
    def parse_valence_csv(lang: Lang):

        if lang is Lang.EN:
            path_to_valence = 'resources/en/wordlists/valences_en_with_liwc.csv'
        Valence.valence_dict = {}
        for vt in ValenceTypeEnum:
            Valence.valence_dict[vt] = {}

        with open(path_to_valence) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=';')
            for i, row in enumerate(csv_reader):
                if i < 2:
                    continue
                row[0] = row[0].strip()                    
                for j, vt in enumerate(ValenceTypeEnum):
                    if len(str(row[j])) > 0 and str(row[j + 1]) != '0':
                        Valence.valence_dict[vt][row[0]] = float(row[j + 1])

    def _compute_value(self, element: TextElement) -> List[float]:
        return sum(1 for word in element.get_words() if word.text in Valence.valence_dict[self.valence_type])
    
    def __repr__(self):
        return f"{self.reduce_function_abbr}({self.abbr}_{self.valence_type.name.lower()} / {self.reduce_depth_abbr})"