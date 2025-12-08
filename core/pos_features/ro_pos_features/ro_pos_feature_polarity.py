from enum import Enum, auto
from rb.core.pos_features.pos_feature import POSFeature
from rb.core.pos_features.ro_pos_features.ro_features_name import RoFeaturesName
from rb.core.lang import Lang
from rb.core.pos import POS
from typing import List
import re


class RoPolarityEnum(Enum):
    NEG = auto()
    POS = auto()

class RoPOSFeaturePolarity(POSFeature):


    _INSTANCE = None

    def __init__(self):
        POSFeature.__init__(self, lang=Lang.RO, name=RoFeaturesName.POL,
                            values=None, pos_supported=None)
        self.values = set([ro_polarity for _, ro_polarity in RoPolarityEnum.__members__.items()])
        self.pos_supported = set([POS.ADV, POS.PART, POS.CCONJ, POS.SCONJ, POS.X, POS.ADP])

    @classmethod
    def get_instance(cls) -> "RoPOSFeaturePolarity":
        if cls._INSTANCE is None:
            cls._INSTANCE = RoPOSFeaturePolarity()
        return cls._INSTANCE

    def get_values(self, tag: str) -> List[RoPolarityEnum]:
        pattern = '(Polarity=[a-zA-Z0-9,]*){1}'
        matches = re.findall(pattern, tag)
        if len(matches) == 0: return []
        nrs = matches[0].split('=')[1].split(',')
        values = []
        for nr in nrs:
            if nr == 'Neg':
                values.append(RoPolarityEnum.NEG)
            elif nr == 'Pos':
                values.append(RoPolarityEnum.POS)
        return values