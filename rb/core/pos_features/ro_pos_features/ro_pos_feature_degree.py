from enum import Enum, auto
from rb.core.pos_features.pos_feature import POSFeature
from rb.core.pos_features.ro_pos_features.ro_features_name import RoFeaturesName
from rb.core.lang import Lang
from rb.core.pos import POS
from typing import List
import re


class RoDegreeEnum(Enum):
    CMP = auto()
    POS = auto()
    SUP = auto()


class RoPOSFeatureDegree(POSFeature):


    _INSTANCE = None
    
    def __init__(self):
        POSFeature.__init__(self, lang=Lang.RO, name=RoFeaturesName.DEG,
                            values=None, pos_supported=None)
        self.values = set([ro_degree for _, ro_degree in RoDegreeEnum.__members__.items()])
        self.pos_supported = set([POS.ADJ, POS.ADV])
    
    @classmethod
    def get_instance(cls) -> "RoPOSFeatureDegree":
        if cls._INSTANCE is None:
            cls._INSTANCE = RoPOSFeatureDegree()
        return cls._INSTANCE

    def get_values(self, tag: str) -> List[RoDegreeEnum]:
        pattern = '(Degree=[a-zA-Z,]*){1}'
        matches = re.findall(pattern, tag)
        if len(matches) == 0: return []
        degrees = matches[0].split('=')[1].split(',')
        values = []
        for deg in degrees:
            if deg == 'Cmp':
                values.append(RoDegreeEnum.CMP)
            elif deg == 'Pos':
                values.append(RoDegreeEnum.POS)
            elif deg == 'Sup':
                values.append(RoDegreeEnum.SUP)
        return values