from threading import Lock
from typing import Callable, List, Tuple, Union

from nltk.corpus import wordnet as wn
from nltk.corpus.reader import CorpusReader, WordNetCorpusReader
from nltk.corpus.reader.wordnet import Synset
from nltk.corpus.util import LazyCorpusLoader

from rb.core.lang import Lang
from rb.core.word import Word
from rb.parser.spacy_parser import SpacyParser

lang_dict = {
    Lang.EN: 'eng',
    Lang.NL: 'nld',
    Lang.FR: 'fra',
    Lang.RO: 'ron',
    Lang.IT: 'ita'
}

class WordNet():

    _instance = None

    def __init__(self):
        self.lock = Lock()
        self.wn = LazyCorpusLoader(
            "wordnet",
            WordNetCorpusReader,
            LazyCorpusLoader("omw-1.4", CorpusReader, r".*/wn-data-.*\.tab", encoding="utf8"))

    @staticmethod
    def get_instance():
        if WordNet._instance is None:
            WordNet._instance = WordNet()
        return WordNet._instance

    def get_synsets(self, word: str, lang: str, pos: str = None) -> List[Synset]:
        self.lock.acquire()
        result = self.wn.synsets(word, pos=pos, lang=lang)
        self.lock.release()
        return result

    def get_synset_hypernyms(self, ss: Synset) -> List[Synset]:
        self.lock.acquire()
        result = ss.hypernyms()
        self.lock.release()
        return result

    def compute_similarity(self, a: str, b: str, lang: Lang, sim: Callable[[Synset, Synset], float]) -> float:
        if lang not in lang_dict:
            return 0
        lang = lang_dict[lang]
        return max([
            sim(syn_a, syn_b) or 0
            for syn_a in self.get_synsets(a, lang=lang)
            for syn_b in self.get_synsets(b, lang=lang)
            if syn_a.pos() == syn_b.pos()],
            default=0)

    def path_similarity(self, a: str, b: str, lang: Lang) -> float:
        return self.compute_similarity(a, b, lang, wn.path_similarity)

    def leacock_chodorow_similarity(self, a: str, b: str, lang: Lang) -> float:
        return self.compute_similarity(a, b, lang, self.wn.lch_similarity)

    def wu_palmer_similarity(self, a: str, b: str, lang: Lang) -> float:
        return self.compute_similarity(a, b, lang, self.wn.wup_similarity)

    def get_synonyms(self, word: Union[str, Word], lang: Lang = None, pos: str = None) -> List[str]:
        if isinstance(word, Word):
            pos = word.pos.to_wordnet()
            lang = word.lang
            word = word.lemma
        if lang not in lang_dict:
            return []
        return list({other
            for ss in self.get_synsets(word, pos=pos, lang=lang_dict[lang]) 
            for other in ss.lemma_names(lang=lang_dict[lang]) 
            if SpacyParser.get_instance().is_dict_word(other, lang)})

    def are_synonyms(self, first: Union[str, Word], second: Union[str, Word], lang: Lang = None, pos: str = None) -> bool:
        if isinstance(first, Word):
            pos = first.pos.to_wordnet()
            lang = first.lang
            first = first.lemma
        if isinstance(second, Word):
            if second.pos.to_wordnet() !=  pos and pos:
                return False 
            second = second.lemma
        if lang not in lang_dict:
            return False
        first_synset = set(self.get_synsets(first, pos=pos, lang=lang_dict[lang]))
        second_synset = set(self.get_synsets(second, pos=pos, lang=lang_dict[lang]))
        return len(first_synset & second_synset) > 0

    def are_hypernyms(self, first: Union[str, Word], second: Union[str, Word], lang: Lang = None, pos: str = None) -> bool:
        if isinstance(first, Word):
            pos = first.pos.to_wordnet()
            lang = first.lang
            first = first.lemma
        if isinstance(second, Word):
            if second.pos.to_wordnet() !=  pos and pos:
                return False 
            second = second.lemma
        if lang not in lang_dict:
            return False
        first_synset = set(self.get_synsets(first, pos=pos, lang=lang_dict[lang]))
        second_synset = set(self.get_synsets(second, pos=pos, lang=lang_dict[lang]))
        first_hypernyms = {hypernym for synset in first_synset for hypernym in self.get_synset_hypernyms(synset)}
        second_hypernyms = {hypernym for synset in second_synset for hypernym in self.get_synset_hypernyms(synset)}
        return len(first_synset & second_hypernyms) > 0 or len(second_synset & first_hypernyms) > 0

    def are_siblings(self, first: Union[str, Word], second: Union[str, Word], lang: Lang = None, pos: str = None) -> bool:
        if isinstance(first, Word):
            pos = first.pos.to_wordnet()
            lang = first.lang
            first = first.lemma
        if isinstance(second, Word):
            if second.pos.to_wordnet() !=  pos and pos:
                return False 
            second = second.lemma
        if lang not in lang_dict:
            return False
        first_synset = set(self.get_synsets(first, pos=pos, lang=lang_dict[lang]))
        second_synset = set(self.get_synsets(second, pos=pos, lang=lang_dict[lang]))
        first_hypernyms = {hypernym for synset in first_synset for hypernym in self.get_synset_hypernyms(synset)}
        second_hypernyms = {hypernym for synset in second_synset for hypernym in self.get_synset_hypernyms(synset)}
        return len(first_hypernyms & second_hypernyms) > 0


    def get_hypernyms(self, word: Union[str, Word], lang: Lang = None, pos: str = None) -> List[str]:
        if isinstance(word, Word):
            pos = word.pos.to_wordnet()
            lang = word.lang
            word = word.lemma
        if lang not in lang_dict:
            return []
        return list({other
            for ss in self.get_synsets(word, pos=pos, lang=lang_dict[lang]) 
            for parent in self.get_synset_hypernyms(ss)
            for other in parent.lemma_names(lang=lang_dict[lang]) 
            if SpacyParser.get_instance().is_dict_word(other, lang)})

    def get_min_path_to_root(self, ss: Synset, lang: Lang = None, pos: str = None) -> int:
        parents: List[Synset] = self.get_synset_hypernyms(ss)
        shortest_path = float('Inf')
        for parent in parents:
            path_length = 1 + self.get_min_path_to_root(parent, lang=lang, pos=pos)
            if  path_length < shortest_path:
                shortest_path = path_length
        
        if shortest_path == float('Inf'):
            return 0
        return shortest_path

    def get_all_paths_lengths_to_root(self, word: Union[str, Word], lang: Lang = None, pos: str = None) -> List[int]:
        if isinstance(word, Word):
            pos = word.pos.to_wordnet()
            lang = word.lang
            word = word.lemma

        if lang not in lang_dict:
            return []
            
        paths = []
        for ss in self.get_synsets(word, pos=pos, lang=lang_dict[lang]):
            path_length = self.get_min_path_to_root(ss, lang=lang, pos=pos)
            paths.append(path_length)
        
        return paths
if __name__ == "__main__":
    print(WordNet.get_instance().path_similarity('hond', 'kat', 'nl'))
    print(WordNet.get_instance().get_hypernyms('animal', Lang.RO))
