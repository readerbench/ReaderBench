from rb.complexity.complexity_index import ComplexityIndex
from rb.core.lang import Lang
from rb.core.text_element import TextElement
from rb.complexity.index_category import IndexCategory
from rb.complexity.measure_function import MeasureFunction
from rb.core.text_element_type import TextElementType
from rb.complexity.discourse.conn_type_enum import ConnTypeEnum
from typing import List, Callable, Set
from rb.utils.rblogger import Logger
from typing import Dict, List
logger = Logger.get_logger()


class Connector(ComplexityIndex):
    
    
    conn_dict: Dict[ConnTypeEnum, List[str]] = None
    
    def __init__(self, lang: Lang, conn_type: ConnTypeEnum,
            reduce_depth: int, reduce_function: MeasureFunction):

        ComplexityIndex.__init__(self, lang=lang, category=IndexCategory.DISCOURSE,
                                 abbr="Connector", reduce_depth=reduce_depth,
                                 reduce_function=reduce_function)
        if Connector.conn_dict is None:
            Connector.parse_connector_list(lang)

        self.conn_type = conn_type


    @staticmethod
    def parse_connector_list(lang: Lang):

        if lang is Lang.RO:
            path_to_wordlist = 'resources/ro/wordlists/connectives_ro.txt'
        elif lang is Lang.EN:
            path_to_wordlist = 'resources/en/wordlists/connectives_en.txt'

        with open(path_to_wordlist, 'rt', encoding='utf-8') as f:
            Connector.conn_dict = {}
            for ct in ConnTypeEnum:
                Connector.conn_dict[ct] = []
            
            conn_type = None
            if lang is Lang.EN:
                for i, line in enumerate(f):
                    if line.find('coordinating_connectives') != -1:
                        conn_type = ConnTypeEnum.COORD
                    elif line.find('logical_connectors') != -1:
                        conn_type = ConnTypeEnum.LOGIC
                    elif line.find('semi_coordinators') != -1:
                        conn_type = ConnTypeEnum.SEMI
                    elif line.find('quasi_coordinators') != -1:
                        conn_type = ConnTypeEnum.QUASI
                    elif line.find('conjunctions') != -1:
                        conn_type = ConnTypeEnum.CONJ
                    elif line.find('disjunctions') != -1:
                        conn_type = ConnTypeEnum.DISJ
                    elif line.find('simple_subordinators') != -1:
                        conn_type = ConnTypeEnum.SIMPLE
                    elif line.find('complex_subordinators') != -1:
                        conn_type = ConnTypeEnum.COMPLEX
                    elif line.find('coordinating_conjuncts') != -1:
                        conn_type = ConnTypeEnum.CONJ
                    elif line.find('addition') != -1:
                        conn_type = ConnTypeEnum.ADDITION
                    elif line.find('contrasts') != -1:
                        conn_type = ConnTypeEnum.CONTRAST
                    elif line.find('sentence_linking') != -1:
                        conn_type = ConnTypeEnum.LINK
                    elif line.find('order') != -1:
                        conn_type = ConnTypeEnum.ORDER
                    elif line.find('reference') != -1:
                        conn_type = ConnTypeEnum.REFERENCE
                    elif line.find('reason_and_purpose') != -1:
                        conn_type = ConnTypeEnum.REASONANDPURPOSE
                    elif line.find('conditions') != -1:
                        conn_type = ConnTypeEnum.COND
                    elif line.find('concessions') != -1:
                        conn_type = ConnTypeEnum.CONCC
                    elif line.find('oppositions') != -1:
                        conn_type = ConnTypeEnum.OPPOSIT
                    elif line.find('temporal_connectors') != -1:
                        conn_type = ConnTypeEnum.TEMPORAL
                    elif line.find('conjuncts') != -1:
                        conn_type = ConnTypeEnum.CONJUNCTS
                    elif len(line.strip()) == 0:
                        continue
                    elif conn_type is not None:
                        Connector.conn_dict[conn_type].append(line.strip())
            elif lang is Lang.RO:
                for i, line in enumerate(f):
                    if line.find('conectori_coordonare') != -1:
                        conn_type = ConnTypeEnum.COORD
                    elif line.find('conectori_logici') != -1:
                        conn_type = ConnTypeEnum.LOGIC
                    elif line.find('semi_coordonatori') != -1:
                        conn_type = ConnTypeEnum.SEMI
                    elif line.find('cvasi_coordonatori') != -1:
                        conn_type = ConnTypeEnum.QUASI
                    elif line.find('conjuncții') != -1:
                        conn_type = ConnTypeEnum.CONJ
                    elif line.find('disjuncții') != -1:
                        conn_type = ConnTypeEnum.DISJ
                    elif line.find('subcoordonatori_simpli') != -1:
                        conn_type = ConnTypeEnum.SIMPLE
                    elif line.find('subcoordonatori_complecși') != -1:
                        conn_type = ConnTypeEnum.COMPLEX
                    elif line.find('conjuncții_coordonatoare') != -1:
                        conn_type = ConnTypeEnum.CONJ
                    elif line.find('adăugire') != -1:
                        conn_type = ConnTypeEnum.ADDITION
                    elif line.find('contraste') != -1:
                        conn_type = ConnTypeEnum.CONTRAST
                    elif line.find('propoziții_conectate') != -1:
                        conn_type = ConnTypeEnum.LINK
                    elif line.find('ordine') != -1:
                        conn_type = ConnTypeEnum.ORDER
                    elif line.find('referință') != -1:
                        conn_type = ConnTypeEnum.REFERENCE
                    elif line.find('motiv_scop') != -1:
                        conn_type = ConnTypeEnum.REASONANDPURPOSE
                    elif line.find('concesii') != -1:
                        conn_type = ConnTypeEnum.CONCC
                    elif line.find('opoziții') != -1:
                        conn_type = ConnTypeEnum.OPPOSIT
                    elif line.find('conectori temporali') != -1:
                        conn_type = ConnTypeEnum.TEMPORAL
                    elif line.find('conjuncții') != -1:
                        conn_type = ConnTypeEnum.CONJUNCTS
                    elif len(line.strip()) == 0:
                        continue
                    elif conn_type is not None:
                        Connector.conn_dict[conn_type].append(line.strip())

    def _compute_value(self, element: TextElement) -> int:
        return sum(1 for word in element.get_words() if element.text.lower() in Connector.conn_dict[self.conn_type])
    
    def __repr__(self):
        return f"{self.reduce_function_abbr}({self.abbr}_{self.conn_type.name.lower()} / {self.reduce_depth_abbr})"
