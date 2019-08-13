from enum import Enum, auto
from rb.core.pos_features.pos_feature import POSFeature
from rb.core.pos_features.ro_pos_features.ro_features_name import RoFeaturesName
from rb.core.lang import Lang
from rb.core.pos import POS
from typing import List
import re


class RoTenseEnum(Enum):
    FUT = auto()
    IMP = auto()
    PAST = auto()
    PQP = auto()
    PRES = auto()


class RoPOSFeatureTense(POSFeature):


    _INSTANCE = None

    def __init__(self):
        POSFeature.__init__(self, lang=Lang.RO, name=RoFeaturesName.TENSE,
                            values=None, pos_supported=None)
        self.values = set([ro_tense for _, ro_tense in RoTenseEnum.__members__.items()])
        self.pos_supported = set([POS.PART, POS.AUX, POS.VERB, POS.X])

    @classmethod
    def get_instance(cls) -> "RoPOSFeatureTense":
        if cls._INSTANCE is None:
            cls._INSTANCE = RoPOSFeatureTense()
        return cls._INSTANCE

    def get_values(self, tag: str) -> List[RoTenseEnum]:
        pattern = '(Tense=[a-zA-Z0-9,]*){1}'
        matches = re.findall(pattern, tag)
        if len(matches) == 0: return []
        nrs = matches[0].split('=')[1].split(',')
        values = []
        for nr in nrs:
            if nr == 'Fut':
                values.append(RoTenseEnum.FUT)
            elif nr == 'Imp':
                values.append(RoTenseEnum.IMP)
            elif nr == 'Past':
                values.append(RoTenseEnum.PAST)
            elif nr == 'Pqp':
                values.append(RoTenseEnum.PQP)
            elif nr == 'Pres':
                values.append(RoTenseEnum.PRES)
        return values