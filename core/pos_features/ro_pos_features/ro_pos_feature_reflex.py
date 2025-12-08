from enum import Enum, auto
from rb.core.pos_features.pos_feature import POSFeature
from rb.core.pos_features.ro_pos_features.ro_features_name import RoFeaturesName
from rb.core.lang import Lang
from rb.core.pos import POS
from typing import List
import re


class RoReflexEnum(Enum):
    YES = auto()

class RoPOSFeatureReflex(POSFeature):


    _INSTANCE = None

    def __init__(self):
        POSFeature.__init__(self, lang=Lang.RO, name=RoFeaturesName.RFL,
                            values=None, pos_supported=None)
        self.values = set([ro_reflex for _, ro_reflex in RoReflexEnum.__members__.items()])
        self.pos_supported = set([POS.PRON])

    @classmethod
    def get_instance(cls) -> "RoPOSFeatureReflex":
        if cls._INSTANCE is None:
            cls._INSTANCE = RoPOSFeatureReflex()
        return cls._INSTANCE

    def get_values(self, tag: str) -> List[RoReflexEnum]:
        pattern = '(Reflex=[a-zA-Z0-9,]*){1}'
        matches = re.findall(pattern, tag)
        if len(matches) == 0: return []
        nrs = matches[0].split('=')[1].split(',')
        values = []
        for nr in nrs:
            if nr == 'Yes':
                values.append(RoReflexEnum.YES)
        return values