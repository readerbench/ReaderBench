from enum import Enum, auto
from rb.core.pos_features.pos_feature import POSFeature
from rb.core.pos_features.ro_pos_features.ro_features_name import RoFeaturesName
from rb.core.lang import Lang
from rb.core.pos import POS
from typing import List
import re


class RoDefiniteEnum(Enum):
    DEF = auto()
    IND = auto()


class RoPOSFeatureDefinite(POSFeature):


    _INSTANCE = None

    def __init__(self):
        POSFeature.__init__(self, lang=Lang.RO, name=RoFeaturesName.DEF,
                            values=None, pos_supported=None)
        self.values = set([ro_definite for _, ro_definite in RoDefiniteEnum.__members__.items()])
        self.pos_supported = set([POS.ADJ, POS.DET, POS.NOUN, POS.NUM, POS.PROPN])
    
    @classmethod
    def get_instance(cls) -> "RoPOSFeatureDefinite":
        if cls._INSTANCE is None:
            cls._INSTANCE = RoPOSFeatureDefinite()
        return cls._INSTANCE

    def get_values(self, tag: str) -> List[RoDefiniteEnum]:
        pattern = '(Definite=[a-zA-Z,]*){1}'
        matches = re.findall(pattern, tag)
        if len(matches) == 0: return []
        definites = matches[0].split('=')[1].split(',')
        values = []
        for defi in definites:
            if defi == 'Def':
                values.append(RoDefiniteEnum.DEF)
            elif defi == 'Ind':
                values.append(RoDefiniteEnum.IND)
        return values