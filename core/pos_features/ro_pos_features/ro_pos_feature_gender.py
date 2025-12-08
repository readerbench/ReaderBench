from enum import Enum, auto
from rb.core.pos_features.pos_feature import POSFeature
from rb.core.pos_features.ro_pos_features.ro_features_name import RoFeaturesName
from rb.core.lang import Lang
from rb.core.pos import POS
from typing import List
import re


class RoGenderEnum(Enum):
    FEM = auto()
    MASC = auto()


class RoPOSFeatureGender(POSFeature):


    _INSTANCE = None

    def __init__(self):
        POSFeature.__init__(self, lang=Lang.RO, name=RoFeaturesName.GEN,
                            values=None, pos_supported=None)
        self.values = set([ro_gender for _, ro_gender in RoGenderEnum.__members__.items()])
        self.pos_supported = set([POS.ADJ, POS.DET, POS.NOUN, POS.NUM, POS.PRON, POS.PROPN, POS.VERB, POS.AUX])

    @classmethod
    def get_instance(cls) -> "RoPOSFeatureGender":
        if cls._INSTANCE is None:
            cls._INSTANCE = RoPOSFeatureGender()
        return cls._INSTANCE

    def get_values(self, tag: str) -> List[RoGenderEnum]:
        pattern = '(Gender=[a-zA-Z,]*){1}'
        matches = re.findall(pattern, tag)
        if len(matches) == 0: return []
        genders = matches[0].split('=')[1].split(',')
        values = []
        for gender in genders:
            if gender == 'Fem':
                values.append(RoGenderEnum.FEM)
            elif gender == 'Masc':
                values.append(RoGenderEnum.MASC)
        return values