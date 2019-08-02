from enum import Enum, auto
from rb.core.pos_features.pos_feature import POSFeature
from rb.core.pos_features.ro_pos_features.ro_features_name import RoFeaturesName
from rb.core.lang import Lang
from rb.core.pos import POS
from typing import List
import re


class RoMoodEnum(Enum):
    IMP = auto()
    IND = auto()
    SUB = auto()


class RoPOSFeatureMood(POSFeature):


    _INSTANCE = None
    
    def __init__(self):
        POSFeature.__init__(self, lang=Lang.RO, name=RoFeaturesName.MOOD,
                            values=None, pos_supported=None)
        self.values = set([ro_mood for _, ro_mood in RoMoodEnum.__members__.items()])
        self.pos_supported = set([POS.AUX, POS.PART, POS.VERB])

    @classmethod
    def get_instance(cls) -> "RoPOSFeatureMood":
        if cls._INSTANCE is None:
            cls._INSTANCE = RoPOSFeatureMood()
        return cls._INSTANCE

    def get_values(self, tag: str) -> List[RoMoodEnum]:
        pattern = '(Mood=[a-zA-Z,]*){1}'
        matches = re.findall(pattern, tag)
        if len(matches) == 0: return []
        moods = matches[0].split('=')[1].split(',')
        values = []
        for mood in moods:
            if mood == 'Imp':
                values.append(RoMoodEnum.IMP)
            elif mood == 'Ind':
                values.append(RoMoodEnum.IND)
            elif mood == 'Ind':
                values.append(RoMoodEnum.SUB)
        return values