from enum import Enum, auto
from rb.core.pos_features.pos_feature import POSFeature
from rb.core.pos_features.ro_pos_features.ro_features_name import RoFeaturesName
from rb.core.lang import Lang
from rb.core.pos import POS
from typing import List
import re


class RoForiegnEnum(Enum):
    YES = auto()


class RoPOSFeatureForeign(POSFeature):

    
    _INSTANCE = None

    def __init__(self):
        POSFeature.__init__(self, lang=Lang.RO, name=RoFeaturesName.FGN,
                            values=None, pos_supported=None)
        self.values = set([ro_fgn for _, ro_fgn in RoForiegnEnum.__members__.items()])
        self.pos_supported = set([POS.ADJ, POS.ADP, POS.DET, POS.NOUN, POS.NUM, POS.PROPN, POS.VERB, POS.X])

    @classmethod
    def get_instance(cls) -> "RoPOSFeatureForeign":
        if cls._INSTANCE is None:
            cls._INSTANCE = RoPOSFeatureForeign()
        return cls._INSTANCE

    def get_values(self, tag: str) -> List[RoForiegnEnum]:
        pattern = '(Foreign=[a-zA-Z,]*){1}'
        matches = re.findall(pattern, tag)
        if len(matches) == 0: return []
        fgns = matches[0].split('=')[1].split(',')
        values = []
        for fgn in fgns:
            if fgn == 'Yes':
                values.append(RoForiegnEnum.YES)
        return values