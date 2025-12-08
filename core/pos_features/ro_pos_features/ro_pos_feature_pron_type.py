from enum import Enum, auto
from rb.core.pos_features.pos_feature import POSFeature
from rb.core.pos_features.ro_pos_features.ro_features_name import RoFeaturesName
from rb.core.lang import Lang
from rb.core.pos import POS
from typing import List
import re


class RoPronTypeEnum(Enum):
    ART = auto()
    DEM = auto()
    EMP = auto()
    IND = auto()
    INT = auto()
    REL = auto()
    NEG = auto()
    PRS = auto()
    TOT = auto()


class RoPOSFeaturePronType(POSFeature):


    _INSTANCE = None

    def __init__(self):
        POSFeature.__init__(self, lang=Lang.RO, name=RoFeaturesName.PRONT,
                            values=None, pos_supported=None)
        self.values = set([ro_pron_type for _, ro_pron_type in RoPronTypeEnum.__members__.items()])
        self.pos_supported = set([POS.DET, POS.PRON, POS.ADV, POS.NUM])

    @classmethod
    def get_instance(cls) -> "RoPOSFeaturePronType":
        if cls._INSTANCE is None:
            cls._INSTANCE = RoPOSFeaturePronType()
        return cls._INSTANCE

    def get_values(self, tag: str) -> List[RoPronTypeEnum]:
        pattern = '(PronType=[a-zA-Z0-9,]*){1}'
        matches = re.findall(pattern, tag)
        if len(matches) == 0: return []
        nrs = matches[0].split('=')[1].split(',')
        values = []
        for nr in nrs:
            if nr == 'Art':
                values.append(RoPronTypeEnum.ART)
            elif nr == 'Dem':
                values.append(RoPronTypeEnum.DEM)
            elif nr == 'Emp':
                values.append(RoPronTypeEnum.EMP)
            elif nr == 'Int':
                values.append(RoPronTypeEnum.INT)
            elif nr == 'Ind':
                values.append(RoPronTypeEnum.IND)
            elif nr == 'Rel':
                values.append(RoPronTypeEnum.REL)
            elif nr == 'Neg':
                values.append(RoPronTypeEnum.NEG)
            elif nr == 'Prs':
                values.append(RoPronTypeEnum.PRS)
            elif nr == 'Tot':
                values.append(RoPronTypeEnum.TOT)

        return values