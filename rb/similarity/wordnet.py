from typing import Callable, List, Tuple, Union
from threading import Lock

from nltk.corpus import wordnet as wn
from nltk.corpus.reader.wordnet import Synset
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

SynsetLock = Lock()

def get_synsets(word: str, lang: str, pos: str = None) -> List[Synset]:
    SynsetLock.acquire()
    result = wn.synsets(word, pos=pos, lang=lang)
    SynsetLock.release()
    return result

def get_synset_hypernyms(ss: Synset) -> List[Synset]:
    SynsetLock.acquire()
    result = ss.hypernyms()
    SynsetLock.release()
    return result

def compute_similarity(a: str, b: str, lang: Lang, sim: Callable[[Synset, Synset], float]) -> float:
    if lang not in lang_dict:
        return 0
    lang = lang_dict[lang]
    return max([
        sim(syn_a, syn_b) or 0
        for syn_a in get_synsets(a, lang=lang)
        for syn_b in get_synsets(b, lang=lang)
        if syn_a.pos() == syn_b.pos()],
        default=0)

def path_similarity(a: str, b: str, lang: Lang) -> float:
    return compute_similarity(a, b, lang, wn.path_similarity)

def leacock_chodorow_similarity(a: str, b: str, lang: Lang) -> float:
    return compute_similarity(a, b, lang, wn.lch_similarity)

def wu_palmer_similarity(a: str, b: str, lang: Lang) -> float:
    return compute_similarity(a, b, lang, wn.wup_similarity)

def get_synonyms(word: Union[str, Word], lang: Lang = None, pos: str = None) -> List[str]:
    if isinstance(word, Word):
        pos = word.pos.to_wordnet()
        lang = word.lang
        word = word.lemma
    if lang not in lang_dict:
        return []
    return list({other
        for ss in get_synsets(word, pos=pos, lang=lang_dict[lang]) 
        for other in ss.lemma_names(lang=lang_dict[lang]) 
        if SpacyParser.get_instance().is_dict_word(other, lang)})

def are_synonyms(first: Union[str, Word], second: Union[str, Word], lang: Lang = None, pos: str = None) -> bool:
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
    first_synset = set(get_synsets(first, pos=pos, lang=lang_dict[lang]))
    second_synset = set(get_synsets(second, pos=pos, lang=lang_dict[lang]))
    return len(first_synset & second_synset) > 0

def are_hypernyms(first: Union[str, Word], second: Union[str, Word], lang: Lang = None, pos: str = None) -> bool:
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
    first_synset = set(get_synsets(first, pos=pos, lang=lang_dict[lang]))
    second_synset = set(get_synsets(second, pos=pos, lang=lang_dict[lang]))
    first_hypernyms = {hypernym for synset in first_synset for hypernym in get_synset_hypernyms(synset)}
    second_hypernyms = {hypernym for synset in second_synset for hypernym in get_synset_hypernyms(synset)}
    return len(first_synset & second_hypernyms) > 0 or len(second_synset & first_hypernyms) > 0

def are_siblings(first: Union[str, Word], second: Union[str, Word], lang: Lang = None, pos: str = None) -> bool:
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
    first_synset = set(get_synsets(first, pos=pos, lang=lang_dict[lang]))
    second_synset = set(get_synsets(second, pos=pos, lang=lang_dict[lang]))
    first_hypernyms = {hypernym for synset in first_synset for hypernym in get_synset_hypernyms(synset)}
    second_hypernyms = {hypernym for synset in second_synset for hypernym in get_synset_hypernyms(synset)}
    return len(first_hypernyms & second_hypernyms) > 0


def get_hypernyms(word: Union[str, Word], lang: Lang = None, pos: str = None) -> List[str]:
    if isinstance(word, Word):
        pos = word.pos.to_wordnet()
        lang = word.lang
        word = word.lemma
    if lang not in lang_dict:
        return []
    return list({other
        for ss in get_synsets(word, pos=pos, lang=lang_dict[lang]) 
        for parent in get_synset_hypernyms(ss)
        for other in parent.lemma_names(lang=lang_dict[lang]) 
        if SpacyParser.get_instance().is_dict_word(other, lang)})

def get_min_path_to_root(ss: Synset, lang: Lang = None, pos: str = None) -> int:
    parents: List[Synset] = get_synset_hypernyms(ss)
    shortest_path = float('Inf')
    for parent in parents:
        path_length = 1 + get_min_path_to_root(parent, lang=lang, pos=pos)
        if  path_length < shortest_path:
            shortest_path = path_length
    
    if shortest_path == float('Inf'):
        return 0
    return shortest_path

def get_all_paths_lengths_to_root(word: Union[str, Word], lang: Lang = None, pos: str = None) -> List[int]:
    if isinstance(word, Word):
        pos = word.pos.to_wordnet()
        lang = word.lang
        word = word.lemma

    if lang not in lang_dict:
        return []
        
    paths = []
    for ss in get_synsets(word, pos=pos, lang=lang_dict[lang]):
        path_length = get_min_path_to_root(ss, lang=lang, pos=pos)
        paths.append(path_length)
    
    return paths
if __name__ == "__main__":
    print(path_similarity('hond', 'kat', 'nl'))
    print(get_hypernyms('animal', Lang.RO))
