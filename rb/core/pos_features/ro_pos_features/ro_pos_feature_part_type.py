from enum import Enum, auto
from rb.core.pos_features.pos_feature import POSFeature
from rb.core.pos_features.ro_pos_features.ro_features_name import RoFeaturesName
from rb.core.lang import Lang
from rb.core.pos import POS
from typing import List
import re


class RoPartTypeEnum(Enum):
    INF = auto()


class RoPOSFeaturePartType(POSFeature):


    _INSTANCE = None

    def __init__(self):
        POSFeature.__init__(self, lang=Lang.RO, name=RoFeaturesName.PARTT,
                            values=None, pos_supported=None)
        self.values = set([ro_part_type for _, ro_part_type in RoPartTypeEnum.__members__.items()])
        self.pos_supported = set([POS.PART])

    @classmethod
    def get_instance(cls) -> "RoPOSFeaturePartType":
        if cls._INSTANCE is None:
            cls._INSTANCE = RoPOSFeaturePartType()
        return cls._INSTANCE

    def get_values(self, tag: str) -> List[RoPartTypeEnum]:
        pattern = '(PartType=[a-zA-Z,]*){1}'
        matches = re.findall(pattern, tag)
        if len(matches) == 0: return []
        nrs = matches[0].split('=')[1].split(',')
        values = []
        for nr in nrs:
            if nr == 'Inf':
                values.append(RoPartTypeEnum.INF)
        return values