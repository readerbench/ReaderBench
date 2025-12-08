from rb.complexity.rhythm.syllabified_dict import SyllabifiedDict
from rb.core.lang import Lang

from typing import List
from queue import Queue

import nltk

# Constants declaration
PHONEMIC_CONSONANTS = {"b", "ch", "d", "dh", "f", "g", "hh", "jh", "k", "l", "m", "n", "ng", "p", "r", "s", "sh", "t", "th", "v", "w", "y", "z", "zh"}
PHONEMIC_VOWELS = {"aa", "ae", "ah", "ao", "aw", "ay", "eh", "er", "ey", "ih", "iy", "ow", "oy", "uh", "uw"}


def is_consonant_sound(sound: str) -> bool:
    return sound in PHONEMIC_CONSONANTS


def is_vocalic_sound(sound: str) -> bool:
    return sound in PHONEMIC_VOWELS


def is_primary_stressed(syllable: List[str]) -> bool:
    """
    Checks if a syllables is primary stressed or not.
    :param syllable: represented as a list of phonemes
    :return: True or False
    """
    return True if any(phoneme.endswith('1') for phoneme in syllable) else False


def get_lang_expansion(lang: Lang) -> str:
    if lang == Lang.EN:
        return "english"
    elif lang == Lang.FR:
        return "french"
    elif lang == Lang.RO:
        return "romanian"
    elif lang == Lang.ES:
        return "spanish"
    elif lang == Lang.DE:
        return "german"
    elif lang == Lang.RU:
        return "russian"
    elif lang == Lang.IT:
        return "italian"
    elif lang == Lang.NL:
        return "dutch"
    else:
        raise Exception('Undefined expansion for abbreviation ' + lang.value)


def get_structures_from_symbols(symbols: List[str]):
    """
    This method is used to extract lists of alliteration or assonance
    from a list of symbols (phonemes) identified by indices.
    :param symbols: list of phonemes
    :return: indices
    """
    structures_indices = list()
    skip_range = 3
    q = Queue()

    q.put(0)
    processed = {i: False for i in range(len(symbols))}
    while not q.empty():
        i = q.get()
        if processed[i]:
            continue
        indices = [i]
        processed[i] = True
        for j in range(i + 1, len(symbols)):
            if symbols[j] == symbols[i]:
                indices.append(j)
                processed[j] = True
            else:
                q.put(j)
                for k in range(j + 1, min(j + skip_range, len(symbols))):
                    if symbols[k] == symbols[i]:
                        i = k
                if i < j:
                    break
        if len(indices) > 1:
            structures_indices.append(indices)

    return structures_indices


def get_rhythmic_structure(lang: Lang, unit: List[str]) -> List[int]:
    """
    This method is implemented according to the definition of rhythmic structure
    formulated by Solomon Marcus in 'Poetica Matematica', 1970
    :param lang: language of the unit
    :param unit: list of words
    :return: a list of integers representing the rhythmic structure of the unit
    """

    syllabified_dict = SyllabifiedDict.get_instance(lang)
    stopwords = nltk.corpus.stopwords.words(get_lang_expansion(lang))

    rhythmic_structure = list()
    cnt = 0
    for word in unit:
        word = word.lower()
        # 1. check if the word is stop word
        if word in stopwords:
            cnt += syllabified_dict[word] if word in syllabified_dict else 1
            continue

        # 2. get the syllables of the word with phonemes (if in dictionary)
        syllables = syllabified_dict[word][0] if word in syllabified_dict else None
        if syllables:
            # 3.1 case when the word was found in dictionary
            for syllable in syllables:
                if is_primary_stressed(syllable):
                    rhythmic_structure.append(cnt)
                    cnt = 0
                else:
                    cnt += 1
        else:
            # 3.2 case when the word was not found in dictionary
            # the word is considered to be unstressed
            # count the number of syllables using a different method
            cnt += 1

    if cnt != 0:
        rhythmic_structure.append(cnt)

    return rhythmic_structure
