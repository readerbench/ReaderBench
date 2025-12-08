from enum import Enum, auto
from rb.core.pos_features.pos_feature import POSFeature
from rb.core.pos_features.ro_pos_features.ro_features_name import RoFeaturesName
from rb.core.lang import Lang
from rb.core.pos import POS
from typing import List
import re


class RoAdpTypeEnum(Enum):
    PREP = auto()


class RoPOSFeatureAdpT(POSFeature):


    _INSTANCE = None

    def __init__(self):
        POSFeature.__init__(self, lang=Lang.RO, name=RoFeaturesName.ADPT,
                            values=None, pos_supported=None)
        self.values = set([ro_adp_type for _, ro_adp_type in RoAdpTypeEnum.__members__.items()])
        self.pos_supported = set([POS.ADP, POS.PUNCT, POS.SYM])

    @classmethod
    def get_instance(cls) -> "RoPOSFeatureAdpT":
        if cls._INSTANCE is None:
            cls._INSTANCE = RoPOSFeatureAdpT()
        return cls._INSTANCE

    def get_values(self, tag: str) -> List[RoAdpTypeEnum]:
        pattern = '(Prep=[a-zA-Z,]*){1}'
        matches = re.findall(pattern, tag)
        if len(matches) == 0: return []
        ro_adp_types = matches[0].split('=')[1].split(',')
        values = []
        for ro_adp_type in ro_adp_types:
            if ro_adp_type == 'Prep':
                values.append(RoAdpTypeEnum.PREP)
        return values