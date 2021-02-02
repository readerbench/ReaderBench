from typing import List

from rb.core.lang import Lang
from rb.core.pos import POS
from rb.core.text_element import TextElement
from rb.core.text_element_type import TextElementType
from rb.parser.spacy_parser import SpacyParser
from rb.core.pos_features.pos_feature_extractor import POSFeatureExtractor
from spacy.tokens import Token


class Word(TextElement):

    MODAL_VERBS = {
        Lang.EN: {"can", "may", "must", "shall", "will", "could", "might", "should", "would"},
    }

    def __init__(self, lang: Lang, token: Token,
                 depth: int = TextElementType.WORD.value,
                 container: TextElement = None):

        TextElement.__init__(self, lang=lang, text=token.text,
                             depth=depth, container=container)
        self.lemma = token.lemma_
        self.pos = POS(token.pos_)
        self.detailed_pos = token.tag_
        self.head: "Word"
        self.dep: str = token.dep_
        self.ent_type_: str  = token.ent_type_
        self.ent_type = token.ent_type
        self.ent_id_ = token.ent_id_
        self.ent_id = token.ent_id
        self.is_stop = token.is_stop
        self.is_alpha = token.is_alpha
        self.children: List["Word"] = []
        self.tag: str = token.tag_
        self.index_in_doc: int = token.i
        self.in_coref = False
        self.coref_clusters = []
        self.pos_features = None#POSFeatureExtractor.create(lang).get_pos_features(self.pos, self.tag)

    @classmethod
    def from_str(cls, lang: Lang, text: str, pos: POS = POS.X) -> "Word":
        token = type('Token', (object,), {'text': text, 'lemma_': text, 'pos_': pos.value, 'tag_': pos.value, 
                     'dep_': None, 'ent_type_': None, 'ent_type': None, 'ent_id_': None, 'ent_id': None,
                     'is_alpha': None, 'is_stop': None, 'tag': None, 'i': None})()
        return Word(lang, token, None)
    
    def is_dict_word(self) -> bool:
        return SpacyParser.get_instance().is_dict_word(self.lemma, self.lang)

    def is_content_word(self) -> bool:
        return self.is_dict_word() and self.pos in {POS.ADJ, POS.ADV, POS.NOUN, POS.VERB}
    
    def is_modal_verb(self) -> bool:
        if self.lang not in self.MODAL_VERBS:
            return False
        return self.pos is POS.VERB and self.lemma in self.MODAL_VERBS[self.lang]
            
    def get_sentences(self) -> List["Sentence"]:
        return []

    def __eq__(self, other):
        if isinstance(other, Word):
            return self.index_in_doc == other.index_in_doc and self.lemma == other.lemma and self.pos == other.pos
        return NotImplemented

    def __str__(self):
        return self.text
    
    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        return hash((self.lemma, self.pos, self.index_in_doc))
