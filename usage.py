from rb.core.lang import Lang
from rb.core.document import Document
from rb.complexity.complexity_index import ComplexityIndex, compute_indices
from rb.core.text_element_type import TextElementType
from rb.core.pos import POS
from rb.similarity.wordnet import path_similarity, get_hypernyms, get_all_paths_lengths_to_root, lang_dict
from rb.core.pos_features.pos_feature_extractor import POSFeatureExtractor
from rb.complexity.word.name_entity_enum import NamedEntityONEnum
from rb.similarity.vector_model import VectorModelType, CorporaEnum, VectorModel
from rb.similarity.vector_model_factory import create_vector_model
from rb.cna.cna_graph import CnaGraph
from rb.processings.ro_corrections.ro_correct import correct_text_ro
from nltk.corpus import wordnet as wn
from flask import jsonify
import argparse

txt_eng = """This is a sample document. It Romanian, (him) can contain, multiple sentences and paragraphs and repeating sentencesm Romania.
This is considered a new block (paragraph).
Therefore in total are 3 blocks."""

txt_ro = """S-a născut repede la 1 februarie 1852,[3] în satul Haimanale (care astăzi îi poartă numele), fiind primul născut al lui Luca Ștefan Caragiale și al Ecaterinei Chiriac Karaboas. Conform unor surse, familia sa ar fi fost de origine aromână.[6] Tatăl său, Luca (1812 - 1870), și frații acestuia, Costache și Iorgu, s-au născut la Constantinopol, 
fiind fiii lui Ștefan, un bucătar angajat la sfârșitul anului 1812 de Ioan Vodă Caragea în suita sa."""

log = open('log.log', 'w', encoding='utf-8')

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run a specific task')
    parser.add_argument('--parser_ro', dest='parser_ro', action='store_true', default=False)
    parser.add_argument('--parser_en', dest='parser_en', action='store_true', default=False)
    parser.add_argument('--indices_ro', dest='indices_ro', action='store_true', default=False)
    parser.add_argument('--indices_en', dest='indices_en', action='store_true', default=False)
    parser.add_argument('--wordnet_ro', dest='wordnet_ro', action='store_true', default=False)
    parser.add_argument('--wordnet_en', dest='wordnet_en', action='store_true', default=False)
    parser.add_argument('--ro_correct_anda', dest='ro_correct_anda', action='store_true', default=False)
    parser.add_argument('--pos_features_ro', dest='pos_features_ro', action='store_true', default=False)

    args = parser.parse_args()
    
    if args.parser_ro:
        print('parser for ro: ', file=log)
        doc = Document(lang=Lang.RO, text=txt_ro)
        for word in doc.get_words():
            print(word.lemma, word.is_stop, word.pos, word.ent_type, word.tag, file=log)

    if args.parser_en:
        print('parser for en: ', file=log)
        doc = Document(lang=Lang.EN, text=txt_eng)
        for word in doc.get_words():
            print(word.lemma, word.is_stop, word.pos, word.ent_type, word.tag, file=log)

    if args.wordnet_ro:
        print('wordnet for ro: ', file=log)
        print(POS.NOUN.to_wordnet(), file=log)
        print('hypernyms (for om) ro', get_hypernyms('om', lang=Lang.RO, pos=POS.NOUN.to_wordnet()), file=log)
        print('paths lengths for om', get_all_paths_lengths_to_root('om', lang=Lang.RO), file=log)

    if args.wordnet_en:
        print('wordnet for en: ', file=log)
        print('hypernyms eng for human', get_hypernyms('human', lang=Lang.EN), file=log)

    if args.indices_en:
        doc = Document(lang=Lang.EN, text=txt_eng)
        en_coca_word2vec = create_vector_model(Lang.EN, VectorModelType.from_str("word2vec"), "coca")
        """you can compute indices without the cna graph, but this means 
           some indices won't be computed"""
        cna_graph_en = CnaGraph(docs=doc, models=[en_coca_word2vec])
        for node, importance in cna_graph_en.importance.items():
            print("{}: {}".format(node, importance))
        compute_indices(doc=doc, cna_graph=cna_graph_en)

        print('\n\nindices at doc level (en): \n\n', file=log)
        for key, v in doc.indices.items():
            print(key, v, file=log)

        print('\n\nindices at block level (en): \n\n', file=log)
        for comp in doc.get_blocks():
            for key, v in comp.indices.items():
                print(comp.text, key, v, file=log)

        print('\n\nindices at sent level (en): \n\n', file=log)
        for comp in doc.get_sentences():
                for key, v in comp.indices.items():
                    print(comp.text, key, v, file=log)

        print('\n\nindices at word level (en): \n\n', file=log)
        for comp in doc.get_sentences():
                for key, v in comp.indices.items():
                    print(comp.text, key, v, file=log)
    if args.indices_ro:
        ro_readme_word2vec = create_vector_model(Lang.RO, VectorModelType.from_str('word2vec'), "readme") 
        doc = Document(lang=Lang.RO, text=txt_ro)
        """you can compute indices without the cna graph, but this means 
           some indices won't be computed"""
        cna_graph_ro = CnaGraph(docs=doc, models=[ro_readme_word2vec])
        compute_indices(doc=doc, cna_graph=cna_graph_ro)

        print('\n\nindices at the doc level (ro): \n\n', file=log)
        for key, v in doc.indices.items():
            print(key, v, file=log)

        print('\n\nindices at the block level (ro): \n\n', file=log)
        for comp in doc.get_blocks():
            for key, v in comp.indices.items():
                print(comp.text, key, v, file=log)

        print('\n\nindices at the sent level (ro): \n\n', file=log)
        for comp in doc.get_sentences():
                for key, v in comp.indices.items():
                    print(comp.text, key, v, file=log)

        print('\n\nindices at word level (ro): \n\n', file=log)
        for comp in doc.get_words():
                for key, v in comp.indices.items():
                    print(comp.text, key, v, file=log)

    if args.pos_features_ro:
        POSFeatureExtractor.create(Lang.RO).print_ud_dict('log.log')
    
    if args.ro_correct_anda:
        txt = "Fiind protejate de stratul de gheaţă, apele mai adânci nu îngheaţă până la fund, ci au, sub stratul de gheaţă, temperatura de 4 grade la care viaţa poate continua"
        print(correct_text_ro(txt), file=log)