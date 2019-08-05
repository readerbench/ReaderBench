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
from typing import Dict, List
from enum import Enum

def get_pos_features(pos, tag) -> Dict[Enum, List[Enum]]:
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
        (RoVerbFormEnum, RoPOSFeatureVerbForm.get_instance()) 
    ]
    for feature_type, feature_instance in feature_instances:
        if pos in feature_instance.pos_supported:
            features[feature_type] = feature_instance.get_values(tag)
    return features