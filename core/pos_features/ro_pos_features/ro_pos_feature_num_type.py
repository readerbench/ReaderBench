from enum import Enum, auto
from rb.core.pos_features.pos_feature import POSFeature
from rb.core.pos_features.ro_pos_features.ro_features_name import RoFeaturesName
from rb.core.lang import Lang
from rb.core.pos import POS
from typing import List
import re


class RoNumTypeEnum(Enum):
    CARD = auto()
    ORD = auto()


class RoPOSFeatureNumType(POSFeature):


    _INSTANCE = None
    
    def __init__(self):
        POSFeature.__init__(self, lang=Lang.RO, name=RoFeaturesName.NRT,
                            values=None, pos_supported=None)
        self.values = set([ro_nr_type for _, ro_nr_type in RoNumTypeEnum.__members__.items()])
        self.pos_supported = set([POS.NUM])

    @classmethod
    def get_instance(cls) -> "RoPOSFeatureNumType":
        if cls._INSTANCE is None:
            cls._INSTANCE = RoPOSFeatureNumType()
        return cls._INSTANCE

    def get_values(self, tag: str) -> List[RoNumTypeEnum]:
        pattern = '(NumType=[a-zA-Z,]*){1}'
        matches = re.findall(pattern, tag)
        if len(matches) == 0: return []
        nr_forms = matches[0].split('=')[1].split(',')
        values = []
        for nr_form in nr_forms:
            if nr_form == 'Card':
                values.append(RoNumTypeEnum.CARD)
            elif nr_form == 'Ord':
                values.append(RoNumTypeEnum.ORD)
        return values