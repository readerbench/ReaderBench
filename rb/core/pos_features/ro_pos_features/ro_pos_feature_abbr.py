from enum import Enum, auto
from rb.core.pos_features.pos_feature import POSFeature
from rb.core.pos_features.ro_pos_features.ro_features_name import RoFeaturesName
from rb.core.lang import Lang
from rb.core.pos import POS
from typing import List
import re


class RoAbbrEnum(Enum):
    YES = auto()


class RoPOSFeatureAbbr(POSFeature):


    _INSTANCE = None

    def __init__(self):
        POSFeature.__init__(self, lang=Lang.RO, name=RoFeaturesName.ABBR,
                            values=None, pos_supported=None)
        self.values = set([ro_abbr for _, ro_abbr in RoAbbrEnum.__members__.items()])
        self.pos_supported = set([POS.ADJ, POS.ADV, POS.NOUN, POS.PRON, POS.X])

    @classmethod
    def get_instance(cls) -> "RoPOSFeatureAbbr":
        if cls._INSTANCE is None:
            cls._INSTANCE = RoPOSFeatureAbbr()
        return cls._INSTANCE

    def get_values(self, tag: str) -> List[RoAbbrEnum]:
        pattern = '(Abbr=[a-zA-Z,]*){1}'
        matches = re.findall(pattern, tag)
        if len(matches) == 0: return []
        abbrs = matches[0].split('=')[1].split(',')
        values = []
        for abbr in abbrs:
            if abbr == 'Yes':
                values.append(RoAbbrEnum.YES)
        return values