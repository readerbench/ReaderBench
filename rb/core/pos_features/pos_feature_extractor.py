
from rb.core.lang import Lang
from rb.core.pos import POS
from enum import Enum
from typing import List, Set, Dict


class POSFeatureExtractor():

    
    def __init__(self):
        pass

    """ factory """
    @classmethod
    def create(clas, lang: Lang) -> "RoPOSFeatureExtractor":
        if lang is Lang.RO:
            from rb.core.pos_features.ro_pos_features.ro_pos_feature_extractor import RoPOSFeatureExtractor
            return RoPOSFeatureExtractor.get_instance()
        else:
            return POSFeatureExtractor()

    def create_ud_dict(self) -> Dict[str, Dict[str, List[Dict[Enum, List[Enum]]]]]:
        pass

    def print_ud_dict(self) -> None:
        pass

    def get_pos_features(self, pos: POS, tag: str) -> Dict[Enum, List[Enum]]:
        pass
    
    def get_all_forms_from_lemma(self, lemma: str) -> Dict[str, Dict[Enum, List[Enum]]]:
        pass