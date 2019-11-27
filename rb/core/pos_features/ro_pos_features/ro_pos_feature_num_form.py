from enum import Enum, auto
from rb.core.pos_features.pos_feature import POSFeature
from rb.core.pos_features.ro_pos_features.ro_features_name import RoFeaturesName
from rb.core.lang import Lang
from rb.core.pos import POS
from typing import List
import re


class RoNumFormEnum(Enum):
    DIGIT = auto()
    ROMAN = auto()
    WORD = auto()


class RoPOSFeatureNumForm(POSFeature):


    _INSTANCE = None
    
    def __init__(self):
        POSFeature.__init__(self, lang=Lang.RO, name=RoFeaturesName.NRF,
                            values=None, pos_supported=None)
        self.values = set([ro_nr_form for _, ro_nr_form in RoNumFormEnum.__members__.items()])
        self.pos_supported = set([POS.NUM])

    @classmethod
    def get_instance(cls) -> "RoPOSFeatureNumForm":
        if cls._INSTANCE is None:
            cls._INSTANCE = RoPOSFeatureNumForm()
        return cls._INSTANCE

    def get_values(self, tag: str) -> List[RoNumFormEnum]:
        pattern = '(NumForm=[a-zA-Z,]*){1}'
        matches = re.findall(pattern, tag)
        if len(matches) == 0: return []
        nr_forms = matches[0].split('=')[1].split(',')
        values = []
        for nr_form in nr_forms:
            if nr_form == 'Digit':
                values.append(RoNumFormEnum.DIGIT)
            elif nr_form == 'Roman':
                values.append(RoNumFormEnum.ROMAN)
            elif nr_form == 'Word':
                values.append(RoNumFormEnum.WORD)
        return values