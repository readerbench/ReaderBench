import re
import spacy
import string

from nltk.tokenize import sent_tokenize
from typing import Iterable, Union
from rb.core.lang import Lang
from rb.parser.spacy_parser import SpacyParser, models
from spacy.language import Language

RE_NEWLINE = re.compile(r"\n", re.UNICODE)
RE_TAGS = re.compile(r"<([^>]+)>", re.UNICODE)
RE_WHITESPACE = re.compile(r"(\s)+", re.UNICODE)


class CustomParser(SpacyParser):
    # Here will be the instance stored.
    __instance = None

    @staticmethod
    def get_instance():
        if CustomParser.__instance is None:
            CustomParser()
        return CustomParser.__instance

    def __init__(self):
        if CustomParser.__instance is not None:
            raise Exception("This class is a singleton!")
        else:
            SpacyParser.__init__(self)
            CustomParser.__instance = self

    def tokenize_sentences(self, block: str) -> Iterable[str]:
        block = RE_NEWLINE.sub(" ", block)
        sentences = sent_tokenize(block)
        for sentence in sentences:
            yield sentence

    # def get_model(self, lang: Union[Lang, str]) -> Language:
    #     if isinstance(lang, str):
    #         lang = Lang(lang)
    #     if lang not in self.loaded_models:
    #         nlp = spacy.load(models[lang])
    #         # workaround to fix the bug for stop words in spacy model
    #         nlp.vocab.add_flag(lambda s: s.lower() in nlp.Defaults.stop_words, spacy.attrs.IS_STOP)
    #         self.loaded_models[lang] = nlp
    #     return self.loaded_models[lang]

    def preprocess(self, text: str, lang: str) -> str:
        text = super(CustomParser, self).preprocess(text, lang)
        text = RE_TAGS.sub("", text)
        text = RE_WHITESPACE.sub(" ", text)
        text = text.lstrip(string.punctuation)
        return text.strip()

    def process(self, doc: str, lang: Lang = Lang.EN):
        doc_list = []
        for sentence in self.tokenize_sentences(doc):
            # remove succeeding whitespace characters
            sentence = RE_WHITESPACE.sub(" ", sentence)
            tokens = self.parse(sentence, lang)
            sentence_dict = {
                "text": sentence,
                "words": []
            }
            for word in tokens:
                # ignore stop words and punctuation
                if word.is_stop or word.text in string.punctuation:
                    continue
                wp = {
                    "text": word.text,
                    "lemma": word.lemma_
                }
                sentence_dict["words"].append(wp)
            doc_list.append(sentence_dict)
        return doc_list


def main():

    parser = CustomParser.get_instance()

    doc = """
            Elephants are large mammals of the family Elephantidae
            and the order Proboscidea. Two species are traditionally recognised,
            the African elephant and the Asian elephant. Elephants are scattered
            throughout sub-Saharan Africa, South Asia, and Southeast Asia. Male
            African elephants are the largest extant terrestrial animals. All
            elephants have a long trunk used for many purposes,
            particularly breathing, lifting water and grasping objects. Their
            incisors grow into tusks, which can serve as weapons and as tools
            for moving objects and digging. Elephants' large ear flaps help
            to control their body temperature. Their pillar-like legs can
            carry their great weight. African elephants have larger ears
            and concave backs while Asian elephants have smaller ears
            and convex or level backs.  
        """

    result = parser.process(doc)
    print(result)


if __name__ == "__main__":

    main()
