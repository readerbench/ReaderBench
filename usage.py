from rb.parser.spacy_parser import SpacyParser
from rb.core.lang import Lang
from rb.core.document import Document
from rb.complexity.complexity_index import ComplexityIndex, compute_indices
from rb.core.text_element_type import TextElementType
from rb.core.pos import POS
from rb.similarity.wordnet import path_similarity, get_hypernyms, get_all_paths_lengths_to_root, lang_dict
from rb.complexity.word.name_entity_enum import NamedEntityONEnum
from nltk.corpus import wordnet as wn

txt1 = """This is a sample document. It Romanian, (him) can contain, multiple sentences and paragraphs and repeating sentencesm Romania.
This is considered a new block (paragraph).
Therefore in total are 3 blocks."""

txt2 = """S-a născut repede la 1 februarie 1852,[3] în satul Haimanale (care astăzi îi poartă numele), fiind primul născut al lui Luca Ștefan Caragiale și al Ecaterinei Chiriac Karaboas. Conform unor surse, familia sa ar fi fost de origine aromână.[6] Tatăl său, Luca (1812 - 1870), și frații acestuia, Costache și Iorgu, s-au născut la Constantinopol, 
fiind fiii lui Ștefan, un bucătar angajat la sfârșitul anului 1812 de Ioan Vodă Caragea în suita sa."""

print(NamedEntityONEnum.PERSON.name)
docs_en = Document(Lang.EN, txt1)
parser = SpacyParser.get_instance().get_model(Lang.EN)
parsed = parser(txt1)


"""how to use wordnet for RO"""

# print(POS.NOUN.to_wordnet())
# print('hypernyms ro', get_hypernyms('om', lang=Lang.RO, pos=POS.NOUN.to_wordnet()))
# print('hypernyms eng', get_hypernyms('human', lang=lang_dict[Lang.EN]))
# print('om', get_all_paths_lengths_to_root('om', lang=Lang.RO))
# print('pe', get_all_paths_lengths_to_root('pe', lang=Lang.RO))
"""how to parse text"""
for sent in parsed.sents:
        print(sent, sent.root)


"""how to compute indices"""
compute_indices(docs_en)

print('\n\nindices at the doc level: \n\n')
for key, v in docs_en.indices.items():
    print(docs_en.text, key, v)

print('\n\nindices at the block level: \n\n')
# indices at the block level
for comp in docs_en.components:
    for key, v in comp.indices.items():
        print(comp.text, key, v)


print('\n\nindices at the sent level: \n\n')
# indices at the sentence level
for comp in docs_en.components:
    for comp2 in comp.components:
        for key, v in comp2.indices.items():
            print(comp2.text, key, v)


"""how to use named entity for ro, how to extract content words"""
# print('parsed text, lemmas, content words, ent types, etc.')
# docs_ro = Document(Lang.RO, txt2)
# for comp1 in docs_ro.components:
# 	# comp is para
#     for comp2 in comp1.components:
# 		# comp2 is sent
#         for ent in comp2.entities:
#             print(ent.text, 'x')
#         # key is word
#         for key in comp2.components:
#             print(key.lemma, key.is_stop, key.pos, key.ent_type, key.ent_type_, key.tag, key.is_content_word())
# if you want only tokns you can do:
# for token in docs_ro.get_tokens():
#     print(token.lemma)