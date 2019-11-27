from typing import List, Callable, Dict
import csv
from rb.complexity.complexity_index import ComplexityIndex
from rb.core.lang import Lang
from rb.core.text_element import TextElement
from rb.complexity.index_category import IndexCategory
from rb.complexity.measure_function import MeasureFunction
from rb.complexity.word.aoe_enum import AoeTypeEnum 
from rb.core.text_element_type import TextElementType
from rb.utils.rblogger import Logger

logger = Logger.get_logger()

class Aoe(ComplexityIndex):


    aoe_dict: Dict[AoeTypeEnum, Dict[str, float]] = None

    def __init__(self, lang: Lang, aoe_type: AoeTypeEnum,
            reduce_depth: int, reduce_function: MeasureFunction):

        ComplexityIndex.__init__(self, lang=lang, category=IndexCategory.WORD,
                                 abbr="AoE", reduce_depth=reduce_depth,
                                 reduce_function=reduce_function)
        if Aoe.aoe_dict is None:
            Aoe.parse_aoe_csv(lang)
        self.aoe_type = aoe_type

    @staticmethod
    def parse_aoe_csv(lang: Lang):

        if lang is Lang.EN:
            path_to_aoe = 'resources/en/wordlists/AoE.csv'
        Aoe.aoe_dict = {}
        for at in AoeTypeEnum:
            Aoe.aoe_dict[at] = {}

        with open(path_to_aoe) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for i, row in enumerate(csv_reader):
                if i < 2:
                    continue
                row[0] = row[0].strip()
                if len(str(row[1])) > 0:
                    Aoe.aoe_dict[AoeTypeEnum.INV_AVG][row[0]] = float(row[1])
                if len(str(row[2])) > 0:
                    Aoe.aoe_dict[AoeTypeEnum.INV_LIN_REG_SLOPE][row[0]] = float(row[2])
                if len(str(row[3])) > 0:
                    Aoe.aoe_dict[AoeTypeEnum.IND_ABOVE_THRESH][row[0]] = float(row[3])
                if len(str(row[4])) > 0:
                    Aoe.aoe_dict[AoeTypeEnum.IND_POLY_FIT_ABOVE][row[0]] = float(row[4])
                if len(str(row[5])) > 0:
                    Aoe.aoe_dict[AoeTypeEnum.INFL_POINT_POLY][row[0]] = float(row[5])

    def process(self, element: TextElement) -> float:
        return self.reduce_function(self.compute_above(element))

    def compute_below(self, element: TextElement) -> List[float]:
        if element.is_word() == True:
            if element.text in Aoe.aoe_dict[self.aoe_type]:
                return [Aoe.aoe_dict[self.aoe_type][element.text]]
            else:
                return []
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
    
    def __repr__(self):
        return self.reduce_function_abbr + self.reduce_depth_abbr + self.abbr + "_" + self.aoe_type.name.lower()