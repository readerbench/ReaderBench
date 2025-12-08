from enum import Enum, auto
from rb.core.pos_features.pos_feature import POSFeature
from rb.core.pos_features.ro_pos_features.ro_features_name import RoFeaturesName
from rb.core.lang import Lang
from rb.core.pos import POS
from typing import List
import re


class RoPossEnum(Enum):
    YES = auto()

class RoPOSFeaturePoss(POSFeature):


    _INSTANCE = None

    def __init__(self):
        POSFeature.__init__(self, lang=Lang.RO, name=RoFeaturesName.POSS,
                            values=None, pos_supported=None)
        self.values = set([ro_poss for _, ro_poss in RoPossEnum.__members__.items()])
        self.pos_supported = set([POS.DET, POS.PRON])

    @classmethod
    def get_instance(cls) -> "RoPOSFeaturePoss":
        if cls._INSTANCE is None:
            cls._INSTANCE = RoPOSFeaturePoss()
        return cls._INSTANCE

    def get_values(self, tag: str) -> List[RoPossEnum]:
        pattern = '(Poss=[a-zA-Z0-9,]*){1}'
        matches = re.findall(pattern, tag)
        if len(matches) == 0: return []
        nrs = matches[0].split('=')[1].split(',')
        values = []
        for nr in nrs:
            if nr == 'Yes':
                values.append(RoPossEnum.YES)
        return values