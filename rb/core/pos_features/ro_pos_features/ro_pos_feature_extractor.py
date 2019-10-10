
from rb.core.lang import Lang
from rb.core.pos import POS
from enum import Enum
from typing import List, Set, Dict
from enum import Enum
from rb.core.pos_features.pos_feature_extractor import POSFeatureExtractor
from conllu import parse, parse_incr
from rb.parser.spacy_parser import computePOS
from rb.core.pos_features.ro_pos_features.ro_features_name import RoFeaturesName
from rb.core.pos_features.ro_pos_features.ro_pos_feature_case import RoPOSFeatureCase, RoCaseEnum
from rb.core.pos_features.ro_pos_features.ro_pos_feature_definite import RoPOSFeatureDefinite, RoDefiniteEnum
from rb.core.pos_features.ro_pos_features.ro_pos_feature_degree import RoPOSFeatureDegree, RoDegreeEnum
from rb.core.pos_features.ro_pos_features.ro_pos_feature_abbr import RoPOSFeatureAbbr, RoAbbrEnum
from rb.core.pos_features.ro_pos_features.ro_pos_feature_adp_type import RoPOSFeatureAdpT, RoAdpTypeEnum
from rb.core.pos_features.ro_pos_features.ro_pos_feature_gender import RoPOSFeatureGender, RoGenderEnum
from rb.core.pos_features.ro_pos_features.ro_pos_feature_mood import RoPOSFeatureMood, RoMoodEnum
from rb.core.pos_features.ro_pos_features.ro_pos_feature_number import RoPOSFeatureNumber, RoNumberEnum
from rb.core.pos_features.ro_pos_features.ro_pos_feature_number_psor import RoPOSFeatureNumberP, RoNumberPEnum
from rb.core.pos_features.ro_pos_features.ro_pos_feature_foreign import RoPOSFeatureForeign, RoForiegnEnum
from rb.core.pos_features.ro_pos_features.ro_pos_feature_num_type import RoPOSFeatureNumType, RoNumTypeEnum
from rb.core.pos_features.ro_pos_features.ro_pos_feature_num_form import RoPOSFeatureNumForm, RoNumFormEnum
from rb.core.pos_features.ro_pos_features.ro_pos_feature_part_type import RoPOSFeaturePartType, RoPartTypeEnum
from rb.core.pos_features.ro_pos_features.ro_pos_feature_person import RoPOSFeaturePerson, RoPersonEnum 
from rb.core.pos_features.ro_pos_features.ro_pos_feature_polarity import RoPOSFeaturePolarity, RoPolarityEnum
from rb.core.pos_features.ro_pos_features.ro_pos_feature_position import RoPOSFeaturePosition, RoPositionEnum
from rb.core.pos_features.ro_pos_features.ro_pos_feature_poss import RoPOSFeaturePoss, RoPossEnum
from rb.core.pos_features.ro_pos_features.ro_pos_feature_pron_type import RoPOSFeaturePronType, RoPronTypeEnum
from rb.core.pos_features.ro_pos_features.ro_pos_feature_strength import RoPOSFeatureStrength, RoStrengthEnum
from rb.core.pos_features.ro_pos_features.ro_pos_feature_tense import RoPOSFeatureTense, RoTenseEnum
from rb.core.pos_features.ro_pos_features.ro_pos_feature_variant import RoPOSFeatureVariant, RoVariantEnum
from rb.core.pos_features.ro_pos_features.ro_pos_feature_verb_form import RoPOSFeatureVerbForm, RoVerbFormEnum

from rb.utils.rblogger import Logger
logger = Logger.get_logger() 


class RoPOSFeatureExtractor(POSFeatureExtractor):

    
    _INSTANCE = None
    UD_RO_FILES = [
        "resources/ro/spacy/ud_tags/ro_rrt-ud-dev.conllu",
        "resources/ro/spacy/ud_tags/ro_rrt-ud-test.conllu",
        "resources/ro/spacy/ud_tags/ro_rrt-ud-train.conllu"]

    def __init__(self):
        self.lang = Lang.RO
        self.create_ud_dict()

    @classmethod
    def get_instance(cls) -> "RoPOSFeatureExtractor":
        if cls._INSTANCE is None:
            cls._INSTANCE = RoPOSFeatureExtractor()
        return cls._INSTANCE

    def create_ud_dict(self) -> Dict[str, Dict[str, List[Dict[Enum, List[Enum]]]]]:
        
        logger.info('Starting parsing ud ro dictionary')
        for ud_file in RoPOSFeatureExtractor.UD_RO_FILES:
            data_file = open(ud_file, "r", encoding="utf-8")
            self.all_forms = {} 

            for tokenlist in parse_incr(data_file):
                for token in tokenlist:
                    s = ""
                    if token['feats'] is not None:
                        for k, v in token['feats'].items():
                            s += '{}={}|'.format(k, v)
                    tag = s
                    pos = computePOS(token['xpostag'], lang=Lang.RO)
                    lemma = token['lemma']
                    form = token['form'].lower()
                    features = self.get_pos_features(pos, tag)
                    if lemma not in self.all_forms:
                        self.all_forms[lemma] = {}
                    if form not in self.all_forms[lemma]:
                        self.all_forms[lemma][form] = []
                    if features not in self.all_forms[lemma][form]:
                        self.all_forms[lemma][form].append(features)
        logger.info('Finished parsing ud ro dictionary')
        return self.all_forms

    def print_ud_dict(self, file_name: str) -> None:
        
        file_handle = open(file_name, 'w', encoding='utf-8')
        all_combinations = set()
        for lemma, forms in self.all_forms.items():
            # print(lemma, len(forms), file=file_handle)
            for word_form, features_sup in forms.items():
                # print('  ', word_form, len(features_sup), file=file_handle)
                for features in features_sup:
                    el = []
                    for key, f in features.items():
                        if len(f) > 0:
                            for ff in f:
                                el.append(ff)
                    print(el, file=file_handle)
                    all_combinations.add(frozenset(el))
                        # if f in features:
                        #     print(features[f], file=file_handle)
        print(len(all_combinations), file=file_handle)

    def get_pos_features(self, pos: POS, tag: str) -> Dict[Enum, List[Enum]]:
        features = {}
        feature_instances = [
            (RoAdpTypeEnum, RoPOSFeatureAdpT.get_instance()),
            (RoCaseEnum, RoPOSFeatureCase.get_instance()),
            (RoAbbrEnum, RoPOSFeatureAbbr.get_instance()),
            (RoDefiniteEnum, RoPOSFeatureDefinite.get_instance()),
            (RoDegreeEnum, RoPOSFeatureDegree.get_instance()),
            (RoGenderEnum, RoPOSFeatureGender.get_instance()),
            (RoMoodEnum, RoPOSFeatureMood.get_instance()),
            (RoNumberEnum, RoPOSFeatureNumber.get_instance()),
            (RoNumberPEnum, RoPOSFeatureNumberP.get_instance()),
            (RoForiegnEnum, RoPOSFeatureForeign.get_instance()),
            (RoNumTypeEnum, RoPOSFeatureNumType.get_instance()),
            (RoNumFormEnum, RoPOSFeatureNumForm.get_instance()),
            (RoPartTypeEnum, RoPOSFeaturePartType.get_instance()),
            (RoPersonEnum, RoPOSFeaturePerson.get_instance()),
            (RoPolarityEnum, RoPOSFeaturePolarity.get_instance()),
            (RoPositionEnum, RoPOSFeaturePosition.get_instance()),
            (RoPossEnum, RoPOSFeaturePoss.get_instance()),
            (RoPronTypeEnum, RoPOSFeaturePronType.get_instance()),
            (RoStrengthEnum, RoPOSFeatureStrength.get_instance()),
            (RoTenseEnum, RoPOSFeatureTense.get_instance()),
            (RoVariantEnum, RoPOSFeatureVariant.get_instance()),
            (RoVerbFormEnum, RoPOSFeatureVerbForm.get_instance())]

        for feature_type, feature_instance in feature_instances:
            #if pos in feature_instance.pos_supported:
            features[feature_type] = feature_instance.get_values(tag)
        return features
    
    def get_all_forms_from_lemma(self, lemma: str) -> Dict[str, Dict[Enum, List[Enum]]]:
        return self.all_forms[lemma]

if __name__ == "__main__":
    feature_extractor = RoPOSFeatureExtractor()
    #feature_extractor.print_ud_dict('log.log')

    for form, feats in feature_extractor.get_all_forms_from_lemma(lemma='vedea').items():
        print(form, feats)