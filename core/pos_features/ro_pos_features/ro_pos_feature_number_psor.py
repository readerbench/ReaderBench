from enum import Enum, auto
from rb.core.pos_features.pos_feature import POSFeature
from rb.core.pos_features.ro_pos_features.ro_features_name import RoFeaturesName
from rb.core.lang import Lang
from rb.core.pos import POS
from typing import List
import re


class RoNumberPEnum(Enum):
    PLUR = auto()
    SING = auto()


class RoPOSFeatureNumberP(POSFeature):


    _INSTANCE = None

    def __init__(self):
        POSFeature.__init__(self, lang=Lang.RO, name=RoFeaturesName.NRP,
                            values=None, pos_supported=None)
        self.values = set([ro_nr_p for _, ro_nr_p in RoNumberPEnum.__members__.items()])
        self.pos_supported = set([POS.DET, POS.PRON])

    @classmethod
    def get_instance(cls) -> "RoPOSFeatureNumberP":
        if cls._INSTANCE is None:
            cls._INSTANCE = RoPOSFeatureNumberP()
        return cls._INSTANCE
        
    def get_values(self, tag: str) -> List[RoNumberPEnum]:
        pattern = '(Number\[psor\]=[a-zA-Z,]*){1}'
        matches = re.findall(pattern, tag)
        if len(matches) == 0: return []
        nrs = matches[0].split('=')[1].split(',')
        values = []
        for nr in nrs:
            if nr == 'Plur':
                values.append(RoNumberPEnum.PLUR)
            elif nr == 'Sing':
                values.append(RoNumberPEnum.SING)
        return values