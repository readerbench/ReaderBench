from rb.core.pos_features.ro_pos_features.ro_pos_feature_case import RoPOSFeatureCase, RoCaseEnum
from rb.core.pos_features.ro_pos_features.ro_pos_feature_definite import RoPOSFeatureDefinite, RoDefiniteEnum
from rb.core.pos_features.ro_pos_features.ro_pos_feature_degree import RoPOSFeatureDegree, RoDegreeEnum
from rb.core.pos_features.ro_pos_features.ro_pos_feature_abbr import RoPOSFeatureAbbr, RoAbbrEnum
from rb.core.pos_features.ro_pos_features.ro_pos_feature_adp_type import RoPOSFeatureAdpT, RoAdpTypeEnum
from rb.core.pos_features.ro_pos_features.ro_pos_feature_gender import RoPOSFeatureGender, RoGenderEnum
from rb.core.pos_features.ro_pos_features.ro_pos_feature_mood import RoPOSFeatureMood, RoMoodEnum
from rb.core.pos_features.ro_pos_features.ro_pos_feature_number import RoPOSFeatureNumber, RoNumberEnum
from rb.core.pos_features.ro_pos_features.ro_pos_feature_number_psor import RoPOSFeatureNumberP, RoNumberPEnum
from rb.core.pos_features.ro_pos_features.ro_features_name import RoFeaturesName
from rb.core.pos_features.ro_pos_features.ro_compute_features import get_pos_features

# from rb.core.document import Document
from rb.core.lang import Lang
from rb.core.pos import POS
from rb.parser.spacy_parser import computePOS

from conllu import parse, parse_incr

log = open('log.log', 'w', encoding='utf-8')

"""
    Each token (word) has some features (e.g. gender, degree). 
    There is an enum in ro_features_name with all the supported features.

    Each of these features has its own class in rb.core.pos_features.ro_pos_features and an enum with all possible values
    In this class you can find the POS which supports (might have) that feature.

    TODO Each token (word) has a dict with key=enum_of_the_fature, value=value_of_the_feature
"""
# def test_pos_features():
#     txt2 = "Am fost la magazin să cumpăr suc CE. și biscuiți."
#     docs_ro = Document(Lang.RO, txt2)

#     case_ro = RoPOSFeatureCase()
#     def_ro = RoPOSFeatureDefinite()
#     degree_ro = RoPOSFeatureDegree()
#     abbr_ro = RoPOSFeatureAbbr()
#     adp_ro = RoPOSFeatureAdpT()
#     gender_ro = RoPOSFeatureGender()
#     mood_ro = RoPOSFeatureMood()

#     for token in docs_ro.get_tokens():
#         print(token.text, token.tag, file=log)
#         print(case_ro.get_values(token.tag), file=log)
#         print(def_ro.get_values(token.tag), file=log)
#         print(degree_ro.get_values(token.tag), file=log)
#         print(abbr_ro.get_values(token.tag), file=log)
#         print(adp_ro.get_values(token.tag), file=log)
#         print(gender_ro.get_values(token.tag), file=log)
#         print(mood_ro.get_values(token.tag), file=log)

            

def create_ud_dict():
    data_file = open("ud_ro_data/ud/ro_rrt-ud-dev.conllu", "r", encoding="utf-8")
    features_dict = {
        RoFeaturesName.CASE: RoPOSFeatureCase(),
        RoFeaturesName.DEF: RoPOSFeatureDefinite(),
        RoFeaturesName.DEG: RoPOSFeatureDegree(),
        RoFeaturesName.ABBR: RoPOSFeatureAbbr(),
        RoFeaturesName.ADPT: RoPOSFeatureAdpT(),
        RoFeaturesName.GEN: RoPOSFeatureGender(),
        RoFeaturesName.MOOD: RoPOSFeatureMood(),
        RoFeaturesName.NR: RoPOSFeatureNumber(),
        RoFeaturesName.NRP: RoPOSFeatureNumberP()
    }

    for tokenlist in parse_incr(data_file):
        for token in tokenlist:
            s = ""
            if token['feats'] is not None:
                for k, v in token['feats'].items():
                    s += '{}={}|'.format(k, v)
            tag = s
            pos = computePOS(token['xpostag'], lang=Lang.RO)
            lemma = token['lemma']
            form = token['form']
            print(pos, lemma, form, tag, file=log)
            # for key, val in features_dict.items():   
            #     print(key.name, val.get_values(tag), file=log)
            for key, v in get_pos_features(pos, tag).items():
                print(key, v, file=log)
            # for el in tokenlist.serialize():
            #     print(el)

if __name__ == "__main__":
    create_ud_dict()
    #test_pos_features()
    pass