from rb.parser.spacy_parser import SpacyParser
from rb.core.lang import Lang
from rb.core.document import Document
from rb.complexity.complexity_index import ComplexityIndex, compute_indices
from rb.core.text_element_type import TextElementType
from rb.core.pos import POS
from rb.similarity.wordnet import path_similarity, get_hypernyms, get_all_paths_lengths_to_root, lang_dict
from rb.core.pos_features.pos_feature_extractor import POSFeatureExtractor
from rb.complexity.word.name_entity_enum import NamedEntityONEnum
from rb.cna.cna_graph import CnaGraph
from rb.similarity.vector_model import VectorModel
from rb.similarity.vector_model import VectorModelType
from rb.similarity.word2vec import Word2Vec
from nltk.corpus import wordnet as wn


txt_eng = """This is a sample document. It Romanian, (him) can contain, multiple sentences and paragraphs and repeating sentencesm Romania.
This is considered a new block (paragraph).
Therefore in total are 3 blocks."""

txt_ro = """S-a născut repede la 1 februarie 1852,[3] în satul Haimanale (care astăzi îi poartă numele), fiind primul născut al lui Luca Ștefan Caragiale și al Ecaterinei Chiriac Karaboas. Conform unor surse, familia sa ar fi fost de origine aromână.[6] Tatăl său, Luca (1812 - 1870), și frații acestuia, Costache și Iorgu, s-au născut la Constantinopol, 
fiind fiii lui Ștefan, un bucătar angajat la sfârșitul anului 1812 de Ioan Vodă Caragea în suita sa."""

log = open('log.log', 'w', encoding='utf-8')

if __name__ == "__main__":

    """how to use wordnet for RO"""
    print(POS.NOUN.to_wordnet(), file=log)
    print('hypernyms ro', get_hypernyms('om', lang=Lang.RO, pos=POS.NOUN.to_wordnet()), file=log)
    print('hypernyms eng', get_hypernyms('human', lang=lang_dict[Lang.EN]), file=log)
    print('om', get_all_paths_lengths_to_root('om', lang=Lang.RO), file=log)
    print('pe', get_all_paths_lengths_to_root('pe', lang=Lang.RO), file=log)

    """indices for en (is the same for RO, just change language) """ 
    w2v = Word2Vec('coca', Lang.EN)
    docs_en = Document(Lang.EN, txt_eng)
    CnaGraph(docs_en, w2v)
    compute_indices(docs_en)

    print('\n\nindices at the doc level: \n\n', file=log)
    for key, v in docs_en.indices.items():
        print(docs_en.text, key, v, file=log)

    print('\n\nindices at the block level: \n\n', file=log)
    for comp in docs_en.components:
        for key, v in comp.indices.items():
            print(comp.text, key, v, file=log)


    print('\n\nindices at the sent level: \n\n', file=log)
    for comp in docs_en.components:
        for comp2 in comp.components:
            for key, v in comp2.indices.items():
                print(comp2.text, key, v, file=log)

    """how to use named entity for ro, how to extract content words, separate texts into paragraphs, sent, etc."""
    docs_ro = Document(Lang.RO, txt_ro)
    for comp1 in docs_ro.components:
        # comp is para
        for comp2 in comp1.components:
            # comp2 is sent
            for ent in comp2.entities:
                # print(ent.text, 'x')
                # key is word
                for key in comp2.components:
                    #print(key.lemma, key.is_stop, key.pos, key.ent_type, key.ent_type_, key.tag, key.is_content_word())
                    pass

    """ if you want only tokens (&their properties) you can do """
    docs_ro = Document(Lang.RO, txt_ro)
    
    
    print('block: ', file=log)
    for block in docs_en.get_blocks():
        print(block, 'end block' , file=log)

    print('sents: ', file=log)
    for sent in docs_en.get_sentences():
        print(sent, file=log)

    print('words: ', file=log)
    for token in docs_ro.get_words():
        print(token.lemma, token.pos, token.text, token.tag, token.pos_features, file=log)
    # print(docs_ro.get_words()[0].get_parent_document().get_sentences())
    # POSFeatureExtractor.create(Lang.RO).print_ud_dict('log.log')
    