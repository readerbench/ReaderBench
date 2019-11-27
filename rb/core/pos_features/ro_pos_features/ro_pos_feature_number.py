from enum import Enum, auto
from rb.core.pos_features.pos_feature import POSFeature
from rb.core.pos_features.ro_pos_features.ro_features_name import RoFeaturesName
from rb.core.lang import Lang
from rb.core.pos import POS
from typing import List
import re


class RoNumberEnum(Enum):
    PLUR = auto()
    SING = auto()


class RoPOSFeatureNumber(POSFeature):


    _INSTANCE = None

    def __init__(self):
        POSFeature.__init__(self, lang=Lang.RO, name=RoFeaturesName.NR,
                            values=None, pos_supported=None)
        self.values = set([ro_nr for _, ro_nr in RoNumberEnum.__members__.items()])
        self.pos_supported = set([POS.ADJ, POS.AUX, POS.DET, POS.NOUN, POS.NUM, POS.PRON, POS.PROPN, POS.VERB])

    @classmethod
    def get_instance(cls) -> "RoPOSFeatureNumber":
        if cls._INSTANCE is None:
            cls._INSTANCE = RoPOSFeatureNumber()
        return cls._INSTANCE

    def get_values(self, tag: str) -> List[RoNumberEnum]:
        pattern = '(Number=[a-zA-Z,]*){1}'
        matches = re.findall(pattern, tag)
        if len(matches) == 0: return []
        nrs = matches[0].split('=')[1].split(',')
        values = []
        for nr in nrs:
            if nr == 'Plur':
                values.append(RoNumberEnum.PLUR)
            elif nr == 'Sing':
                values.append(RoNumberEnum.SING)
        return values