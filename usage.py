from rb.core.lang import Lang
from rb.core.document import Document
from rb.complexity.complexity_index import ComplexityIndex, compute_indices
from rb.core.text_element_type import TextElementType
from rb.core.pos import POS
from rb.similarity.wordnet import path_similarity, get_hypernyms, get_all_paths_lengths_to_root, lang_dict
from rb.core.pos_features.pos_feature_extractor import POSFeatureExtractor
from rb.complexity.word.name_entity_enum import NamedEntityONEnum
from rb.similarity.vector_model import VectorModelType, CorporaEnum, VectorModel
from rb.similarity.vector_model_instance import VECTOR_MODELS
from nltk.corpus import wordnet as wn


txt_eng = """This is a sample document. It Romanian, (him) can contain, multiple sentences and paragraphs and repeating sentencesm Romania.
This is considered a new block (paragraph).
Therefore in total are 3 blocks."""

txt_ro = """S-a născut repede la 1 februarie 1852,[3] în satul Haimanale (care astăzi îi poartă numele), fiind primul născut al lui Luca Ștefan Caragiale și al Ecaterinei Chiriac Karaboas. Conform unor surse, familia sa ar fi fost de origine aromână.[6] Tatăl său, Luca (1812 - 1870), și frații acestuia, Costache și Iorgu, s-au născut la Constantinopol, 
fiind fiii lui Ștefan, un bucătar angajat la sfârșitul anului 1812 de Ioan Vodă Caragea în suita sa."""

log = open('log.log', 'w', encoding='utf-8')

if __name__ == "__main__":

    r = 2

    if r == 1:
        """how to use wordnet for RO"""
        print(POS.NOUN.to_wordnet(), file=log)
        print('hypernyms ro', get_hypernyms('om', lang=Lang.RO, pos=POS.NOUN.to_wordnet()), file=log)
        print('hypernyms eng', get_hypernyms('human', lang=lang_dict[Lang.EN]), file=log)
        print('om', get_all_paths_lengths_to_root('om', lang=Lang.RO), file=log)
        print('pe', get_all_paths_lengths_to_root('pe', lang=Lang.RO), file=log)

        """indices for en (is the same for RO, just change language) """ 
        vector_model = VECTOR_MODELS[Lang.EN][CorporaEnum.COCA][VectorModelType.WORD2VEC](
            name=CorporaEnum.COCA.value, lang=Lang.EN)
        doc = Document(lang=Lang.EN, text=txt_eng)

        compute_indices(doc, use_cna_graph=True, vector_models=[vector_model])

        print('\n\nindices at the doc level: \n\n', file=log)
        for key, v in doc.indices.items():
            print(key, v, file=log)

        print('\n\nindices at the block level: \n\n', file=log)
        for comp in doc.get_blocks():
            for key, v in comp.indices.items():
                print(comp.text, key, v, file=log)


        print('\n\nindices at the sent level: \n\n', file=log)
        for comp in doc.get_sentences():
                for key, v in comp.indices.items():
                    print(comp.text, key, v, file=log)
    elif r == 2:
        """ for ro """
        vector_model = VECTOR_MODELS[Lang.RO][CorporaEnum.README][VectorModelType.WORD2VEC](
                        name=CorporaEnum.README, lang=Lang.RO)
        doc = Document(lang=Lang.RO, text=txt_ro)
        compute_indices(doc, use_cna_graph=True, vector_models=[vector_model])

        print('\n\nindices at the doc level: \n\n', file=log)
        for key, v in doc.indices.items():
            print(key, v, file=log)

        print('\n\nindices at the block level: \n\n', file=log)
        for comp in doc.get_blocks():
            for key, v in comp.indices.items():
                print(comp.text, key, v, file=log)

        print('\n\nindices at the sent level: \n\n', file=log)
        for comp in doc.get_sentences():
                for key, v in comp.indices.items():
                    print(comp.text, key, v, file=log)

        for word in doc.get_words():
            print(word, word.idx, word.is_alpha)
    # print(docs_ro.get_words()[0].get_parent_document().get_sentences())
    else:
        POSFeatureExtractor.create(Lang.RO).print_ud_dict('log.log')
    