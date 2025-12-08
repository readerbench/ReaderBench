
from rb.core.lang import Lang
from rb.core.pos import POS
from enum import Enum
from typing import List, Set

""" each pos feature (case, gender, mood, etc.) is defined by this class """

class POSFeature():

    
    def __init__(self, lang: Lang,
                 name: Enum, values: Set[Enum], 
                 pos_supported: Set[POS]):
        self.lang = lang
        self.name = name
        self.values = values
        self.pos_supported = pos_supported

    def get_values(self) -> List:
        pass
    
