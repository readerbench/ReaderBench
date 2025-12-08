from enum import Enum, auto
from rb.core.pos_features.pos_feature import POSFeature
from rb.core.pos_features.ro_pos_features.ro_features_name import RoFeaturesName
from rb.core.lang import Lang
from rb.core.pos import POS
from typing import List
import re


class RoCaseEnum(Enum):
    ACC = auto()
    NOM = auto()
    DAT = auto()
    GEN = auto()
    VOC = auto()


class RoPOSFeatureCase(POSFeature):


    _INSTANCE = None

    def __init__(self):
        POSFeature.__init__(self, lang=Lang.RO, name=RoFeaturesName.CASE,
                            values=None, pos_supported=None)
        self.values = set([ro_case for _, ro_case in RoCaseEnum.__members__.items()])
        self.pos_supported = set([POS.ADP, POS.PRON, POS.ADJ, POS.DET, POS.NOUN, POS.NUM, POS.PROPN])

    @classmethod
    def get_instance(cls) -> "RoPOSFeatureCase":
        if cls._INSTANCE is None:
            cls._INSTANCE = RoPOSFeatureCase()
        return cls._INSTANCE

    def get_values(self, tag: str) -> List[RoCaseEnum]:
        pattern = '(Case=[a-zA-Z,]*){1}'
        matches = re.findall(pattern, tag)
        if len(matches) == 0: return []
        cases = matches[0].split('=')[1].split(',')
        values = []
        for case in cases:
            if case == 'Dat':
                values.append(RoCaseEnum.DAT)
            elif case == 'Acc':
                values.append(RoCaseEnum.ACC)
            elif case == 'Gen':
                values.append(RoCaseEnum.GEN)
            elif case == 'Nom':
                values.append(RoCaseEnum.NOM)
            elif case == 'Voc':
                values.append(RoCaseEnum.VOC)
        return values

