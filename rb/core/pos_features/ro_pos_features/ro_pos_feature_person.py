from enum import Enum, auto
from rb.core.pos_features.pos_feature import POSFeature
from rb.core.pos_features.ro_pos_features.ro_features_name import RoFeaturesName
from rb.core.lang import Lang
from rb.core.pos import POS
from typing import List
import re


class RoPersonEnum(Enum):
    FIRST = auto()
    SECOND = auto()
    THIRD = auto()

class RoPOSFeaturePerson(POSFeature):


    _INSTANCE = None

    def __init__(self):
        POSFeature.__init__(self, lang=Lang.RO, name=RoFeaturesName.PER,
                            values=None, pos_supported=None)
        self.values = set([ro_person for _, ro_person in RoPersonEnum.__members__.items()])
        self.pos_supported = set([POS.AUX, POS.DET, POS.PRON, POS.VERB])

    @classmethod
    def get_instance(cls) -> "RoPOSFeaturePerson":
        if cls._INSTANCE is None:
            cls._INSTANCE = RoPOSFeaturePerson()
        return cls._INSTANCE

    def get_values(self, tag: str) -> List[RoPersonEnum]:
        pattern = '(Person=[a-zA-Z0-9,]*){1}'
        matches = re.findall(pattern, tag)
        if len(matches) == 0: return []
        nrs = matches[0].split('=')[1].split(',')
        values = []
        for nr in nrs:
            if nr == '1':
                values.append(RoPersonEnum.FIRST)
            elif nr == '2':
                values.append(RoPersonEnum.SECOND)
            elif nr == '3':
                values.append(RoPersonEnum.THIRD)
        return values