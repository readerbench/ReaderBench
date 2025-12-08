from enum import Enum, auto
from rb.core.pos_features.pos_feature import POSFeature
from rb.core.pos_features.ro_pos_features.ro_features_name import RoFeaturesName
from rb.core.lang import Lang
from rb.core.pos import POS
from typing import List
import re


class RoVerbFormEnum(Enum):
    FIN = auto()
    GER = auto()
    INF = auto()
    PART = auto()


class RoPOSFeatureVerbForm(POSFeature):


    _INSTANCE = None

    def __init__(self):
        POSFeature.__init__(self, lang=Lang.RO, name=RoFeaturesName.VBF,
                            values=None, pos_supported=None)
        self.values = set([ro_verb_form for _, ro_verb_form in RoVerbFormEnum.__members__.items()])
        self.pos_supported = set([POS.AUX, POS.VERB, POS.X])

    @classmethod
    def get_instance(cls) -> "RoPOSFeatureVerbForm":
        if cls._INSTANCE is None:
            cls._INSTANCE = RoPOSFeatureVerbForm()
        return cls._INSTANCE

    def get_values(self, tag: str) -> List[RoVerbFormEnum]:
        pattern = '(VerbForm=[a-zA-Z0-9,]*){1}'
        matches = re.findall(pattern, tag)
        if len(matches) == 0: return []
        nrs = matches[0].split('=')[1].split(',')
        values = []
        for nr in nrs:
            if nr == 'Fin':
                values.append(RoVerbFormEnum.FIN)
            elif nr == 'Ger':
                values.append(RoVerbFormEnum.GER)
            elif nr == 'Inf':
                values.append(RoVerbFormEnum.INF)
            elif nr == 'Part':
                values.append(RoVerbFormEnum.PART)
        return values