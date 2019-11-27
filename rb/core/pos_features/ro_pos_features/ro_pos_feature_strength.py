from enum import Enum, auto
from rb.core.pos_features.pos_feature import POSFeature
from rb.core.pos_features.ro_pos_features.ro_features_name import RoFeaturesName
from rb.core.lang import Lang
from rb.core.pos import POS
from typing import List
import re


class RoStrengthEnum(Enum):
    STRONG = auto()
    WEAK = auto()

class RoPOSFeatureStrength(POSFeature):


    _INSTANCE = None

    def __init__(self):
        POSFeature.__init__(self, lang=Lang.RO, name=RoFeaturesName.STRN,
                            values=None, pos_supported=None)
        self.values = set([ro_strength for _, ro_strength in RoStrengthEnum.__members__.items()])
        self.pos_supported = set([POS.PRON, POS.X])

    @classmethod
    def get_instance(cls) -> "RoPOSFeatureStrength":
        if cls._INSTANCE is None:
            cls._INSTANCE = RoPOSFeatureStrength()
        return cls._INSTANCE

    def get_values(self, tag: str) -> List[RoStrengthEnum]:
        pattern = '(Strength=[a-zA-Z0-9,]*){1}'
        matches = re.findall(pattern, tag)
        if len(matches) == 0: return []
        nrs = matches[0].split('=')[1].split(',')
        values = []
        for nr in nrs:
            if nr == 'Strong':
                values.append(RoStrengthEnum.STRONG)
            elif nr == 'Weak':
                values.append(RoStrengthEnum.WEAK)
        return values