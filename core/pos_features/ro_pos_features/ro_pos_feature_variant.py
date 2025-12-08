from enum import Enum, auto
from rb.core.pos_features.pos_feature import POSFeature
from rb.core.pos_features.ro_pos_features.ro_features_name import RoFeaturesName
from rb.core.lang import Lang
from rb.core.pos import POS
from typing import List
import re


class RoVariantEnum(Enum):
    SHORT = auto()


class RoPOSFeatureVariant(POSFeature):


    _INSTANCE = None

    def __init__(self):
        POSFeature.__init__(self, lang=Lang.RO, name=RoFeaturesName.VAR,
                            values=None, pos_supported=None)
        self.values = set([ro_variant for _, ro_variant in RoVariantEnum.__members__.items()])
        self.pos_supported = set([POS.ADJ, POS.ADP, POS.ADV, POS.AUX, POS.CONJ, POS.DET, 
            POS.NOUN, POS.PART, POS.PRON, POS.SCONJ, POS.VERB, POS.X])

    @classmethod
    def get_instance(cls) -> "RoPOSFeatureVariant":
        if cls._INSTANCE is None:
            cls._INSTANCE = RoPOSFeatureVariant()
        return cls._INSTANCE

    def get_values(self, tag: str) -> List[RoVariantEnum]:
        pattern = '(Variant=[a-zA-Z0-9,]*){1}'
        matches = re.findall(pattern, tag)
        if len(matches) == 0: return []
        nrs = matches[0].split('=')[1].split(',')
        values = []
        for nr in nrs:
            if nr == 'Short':
                values.append(RoVariantEnum.SHORT)
        return values