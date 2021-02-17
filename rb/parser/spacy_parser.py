import logging
import re
from typing import Dict, Iterable, List, Union

import neuralcoref
import spacy
from nltk.tokenize import sent_tokenize
from rb.core.lang import Lang
from rb.core.pos import POS
from rb.utils.downloader import check_spacy_version, download_spacy_model
from rb.utils.rblogger import Logger
from spacy.lang.ro import Romanian
from spacy.language import Language
from spacy.tokens import Doc, Span, Token

from sentence_splitter import SentenceSplitter

# JSON Example localhost:8081/spacy application/json
# {
#     "lang" : "en",
#     "blocks" : ["După terminarea oficială a celui de-al doilea război mondial, în conformitate cu discursul lui W. Churchill (prim ministru al Regatului Unit la acea dată), de la Fulton, s-a declanșat Războiul rece și a apărut conceptul de cortină de fier. Urmare a politicii consecvente de apărare a sistemului economic și politic (implicit a intereslor economice ale marelui capital din lumea occidentală) trupele germane, în calitate de prizonieri, aflate pe teritoriul Germaniei de Vest au fost reînarmate și au constituit baza viitorului Bundeswehr - armata regulată a R.F.G."]
# }

logger = Logger.get_logger()

models = {
    Lang.EN: 'en_core_web_lg',
    Lang.NL: 'nl_core_news_lg',
    Lang.FR: 'fr_core_news_lg',
    Lang.ES: 'es_core_news_lg',
    Lang.DE: 'de_core_news_lg',
    Lang.IT: 'it_core_news_lg',
    Lang.RO: 'ro_core_news_lg',
    Lang.RU: 'ru_ud_ft',
}

custom_models = {Lang.RU}

splitters = {
    # Lang.RO: SentenceSplitter('ro'),
    Lang.RU: SentenceSplitter('ru'),
}

normalization = {
    'ro': [
        (re.compile("ş"), "ș"),
        (re.compile("Ş"), "Ș"),
        (re.compile("ţ"), "ț"),
        (re.compile("Ţ"), "Ț"),
        (re.compile("(\w)î(\w)"), "\g<1>â\g<2>")
    ]
}

re_missing_space = re.compile("([a-z]+)\.([A-Z][a-z]+)")

def convertToPenn(pos: str, lang: Lang) -> str:
    if lang == Lang.FR:
        pos = pos.lower()
        if pos.startswith('noun') or pos.startswith('propn'):
            return "NN"
        if pos.startswith("verb"):
            return "VB"
        if pos.startswith("adj"):
            return "JJ"
        if pos.startswith("adv"):
            return "RB"
        if pos.startswith("adp"):
            return "IN"
        if pos.startswith("cconj"):
            return "CC"
        return ""
    if lang == Lang.NL:
        pos = pos.lower()
        if pos.startswith('n_') or pos.startswith('n|') or pos.startswith('propn'):
            return "NN"
        if pos.startswith("v_") or pos.startswith("v|"):
            return "VB"
        if pos.startswith("adj"):
            return "JJ"
        if pos.startswith("adv"):
            return "RB"
        if pos.startswith("adp"):
            return "IN"
        if pos.startswith("cconj") or pos.startswith("conj"):
            return "CC"
        return ""
    if lang == Lang.RO: 
        pos = pos.lower()
        if pos.startswith("n"):
            return "NN"
        if pos.startswith("v"):
            return "VB"
        if pos.startswith("a"):
            return "JJ"
        if pos.startswith("r"):
            return "RB"
        if pos.startswith("s") or pos.startswith("cs"):
            return "IN"
        if pos.startswith("c"):
            return "CC"
        return ""
    if len(pos) > 2:
        return pos[:2]
    return pos
    
def computePOS(token: Union[str, Token], lang: Lang) -> POS:
    if lang == Lang.RO:
        if isinstance(token, Token):
            pos = token.tag_.lower()
        else:
            pos = token.lower()
        if pos.startswith("comma") or pos.startswith("period"):
            return POS.PUNCT
        if pos.startswith("n"):
            return POS.NOUN
        if pos.startswith("v"):
            return POS.VERB
        if pos.startswith("a"):
            return POS.ADJ
        if pos.startswith("r"):
            return POS.ADV
        if pos.startswith("s") or pos.startswith("cs"):
            return POS.ADP
        if pos.startswith("c"):
            return POS.CCONJ
        if pos.startswith("m"):
            return POS.NUM
        if pos.startswith("t"):
            return POS.DET
        if pos.startswith("d"):
            return POS.PRON
        return POS.X
    try:
        return POS(token.pos_)
    except:
        return POS.X

class SpacyParser:

    _INSTANCE = None

    def __init__(self):
        self.pipelines = {
            lang: spacy.util.get_lang_class(lang.value)()
            for lang in models
        }
        # for pipeline in self.pipelines.values():
        #     component = pipeline.create_pipe('tagger')   # 3. create the pipeline components
        #     pipeline.add_pipe(component)
        self.loaded_models = {}
    
    @classmethod
    def get_instance(cls) -> "SpacyParser":
        if cls._INSTANCE is None:
            cls._INSTANCE = SpacyParser()
        return cls._INSTANCE
        
    def preprocess(self, text: str, lang: str) -> str:
        if lang not in normalization:
            return text
        for pattern, replacement in normalization[lang]:
            text = re.sub(pattern, replacement, text)
        return text

    def get_tokens_lemmas(self, sentences: Iterable, lang: str) -> Iterable:
        if lang not in self.pipelines:
            return None
        pipeline = self.pipelines[lang]
        # sbd = pipeline.create_pipe('sentencizer')
        # pipeline.add_pipe(sbd)
        doc = pipeline.pipe((sent[:1].lower() + sent[1:] for sent in sentences), batch_size=100000, n_threads=16)
        # print([sent.string.strip() for sent in doc.sents])
        # print(len(doc.sents))
        # print("====================")
        # for token in doc:
        #     print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
        #   token.shape_, token.is_alpha, token.is_stop)
        # print("====================")
        return doc
        # return [(token.text, token.lemma_) for token in doc]

    def parse_block(self, block: str, lang: Lang) -> List[Span]:
        block = re.sub(re_missing_space, r"\1. \2", block)
        if lang in custom_models:
            return [self.parse(sent, lang) for sent in splitters[lang].split(block)]
        else:
            doc = self.parse(block, lang)
            return [sent for sent in doc.sents]

    def tokenize_sentences(self, block: str) -> List[str]:
        return sent_tokenize(block)
    
    @staticmethod
    def line_splitter(doc):
        for token in doc:
            if "\n" in token.text_with_ws and token.i < len(doc) - 1:
                token.nbor().sent_start = True
        return doc


    def get_model(self, lang: Union[Lang, str]) -> Language:
        if isinstance(lang, str):
            lang = Lang(lang)
        if lang not in self.loaded_models:
            if lang not in custom_models:
                try:
                    self.loaded_models[lang] = spacy.load(models[lang])
                    self.loaded_models[lang].add_pipe(SpacyParser.line_splitter, name='sentence_segmenter', before='parser')
                    if lang is Lang.EN:
                        neuralcoref.add_to_pipe(self.loaded_models[lang])
                except:
                    logger.error("spaCy model not found. You can download it by running the following commmand:\nsudo python3 -m spacy download {}".format(models[lang]))
                    return None
            else:
                folder = "resources/{}/spacy/{}".format(lang.value, models[lang])
                if check_spacy_version(lang, models[lang]):
                    if not download_spacy_model(lang, models[lang]):
                        logger.error("Could not download model for lang {}".format(lang.value))
                        return None
                self.loaded_models[lang] = spacy.load(folder)
        return self.loaded_models[lang]

    def parse(self, sentence: str, lang: Lang) -> Doc:
        model = self.get_model(lang)
        if model is None:
            raise Exception("Model not loaded")
        doc = model(sentence)
        for token in doc:
            token.pos_ = computePOS(token, lang).value
        return doc


    def is_dict_word(self, word: str, lang: Union[Lang, str]) -> bool:
        model = self.get_model(lang)
        return model.vocab.has_vector(word)

if __name__ == "__main__":
    spacyInstance = SpacyParser()

    sent = """
        După terminarea oficială a celui de-al doilea război mondial, în conformitate cu discursul lui W. Churchill (prim ministru al Regatului Unit la acea dată), de la Fulton, s-a declanșat Războiul rece și a apărut conceptul de cortină de fier. Urmare a politicii consecvente de apărare a sistemului economic și politic (implicit a intereslor economice ale marelui capital din lumea occidentală) trupele germane, în calitate de "prizonieri", aflate pe teritoriul Germaniei de Vest au fost reînarmate și au constituit baza viitorului "Bundeswehr" - armata regulată a R.F.G.

        Pe fondul evenimentelor din 1948 din Cehoslovacia (expulzări ale etnicilor germani, alegeri, reconstrucție economică) apare infiltrarea agenților serviciilor speciale ale S.U.A. și Marii Britanii cu rol de "agitatori". Existând cauza, trupele sovietice nu părăsesc Europa Centrală și de Est cucerită-eliberată, staționând pe teritoriul mai multor state. Aflate pe linia de demarcație dintre cele două blocuri foste aliate, armata sovietică nu a plecat din Ungaria decât după dizolvarea Tratatului de la Varșovia.
        """

    # sent = """
    #     După terminarea oficială a celui de-al doilea război mondial, în conformitate cu discursul lui Churchill, de la Fulton, s-a declanșat Războiul rece și a apărut conceptul de cortină de fier."""

    # print(spacyInstance.get_tokens_lemmas(sent))
    # doc = spacyInstance.parse("My sister has a dog. She loves him.", 'en')
    doc = spacyInstance.parse("Pensée des enseignants, production d’écrits, ingénierie éducative, enseignement à distance, traitement automatique de la langue, outils cognitifs, feedback automatique", 'fr')
    for token in doc:
        print(convertToPenn(token.tag_, 'fr'))   
    # print(spacyInstance.preprocess("coborî", 'ro'))
