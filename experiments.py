import csv
from collections import deque
from typing import Dict, List

from rb.core.document import Document
from rb.core.lang import Lang
from rb.core.pos import POS
from rb.core.word import Word
from rb.core.sentence import Sentence


def get_features(word: Word) -> Dict[str, str]:
    return dict([tuple(feature.split("=")) for feature in word.tag.split("__")[1].split("|")])

def is_infinitive(word: Word) -> bool:
    if not word.tag.startswith('VERB'):
        return False
    features = get_features(word)
    return features["VerbForm"] == "Inf"

def is_gerund(word: Word) -> bool:
    if not word.tag.startswith('VERB'):
        return False
    features = get_features(word)
    return features["VerbForm"] == "Conv"

def is_negative(word: Word) -> bool:
    if not word.text.lower() == "не" or not word.dep == 'advmod':
        return False
    verb = word.head
    return verb.pos is  POS.VERB

def is_nmod(word: Word) -> bool:
    if word.pos is not POS.NOUN or not word.dep == 'nmod':
        return False
    return word.head.pos is POS.NOUN

def is_part_noun(word: Word) -> bool:
    if not word.dep == 'acl':
        return False
    return word.index_in_doc < word.head.index_in_doc

def is_noun_inf(word: Word) -> bool:
    if word.pos is not POS.VERB or not word.dep == 'obl':
        return False
    features = get_features(word)
    return features["VerbForm"] == "Inf" and word.head.pos is POS.NOUN and word.index_in_doc > word.head.index_in_doc
    
def is_attr_clause(word: Word) -> bool:
    if word.pos is not POS.PRON or not word.dep == 'nsubj':
        return False
    verb = word.head
    if verb.pos is not POS.VERB or not verb.dep == 'acl:relcl':
        return False
    noun = verb.head
    return noun.pos is POS.NOUN and noun.index_in_doc < word.index_in_doc < verb.index_in_doc
    
def is_MSVO(sentence: Sentence) -> bool:
    a = -1
    s = -1
    v = sentence.root.index_in_doc
    o = -1
    for c in sentence.root.children:
        if c.dep == "obl":
            a = c.index_in_doc
        if c.dep == "nsubj":
            s = c.index_in_doc
        if c.dep == "obj":
            o = c.index_in_doc
    return a < s < v < o

def bfs(word: Word) -> List[Word]:
    q = deque()
    q.append(word)
    visited = set()
    result = []
    while len(q) > 0:
        x = q.popleft()
        if x in visited:
            continue
        visited.add(x)
        result.append(x)
        for y in x.children:
            if y not in visited:
                q.append(y)
    return list(sorted(result, key=lambda x: x.index_in_doc))
                
        

if __name__ == "__main__":
    with open("russian/51.csv", "rt", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=';', quotechar='"')
        for text, output, _, _ in reader:
            doc = Document(Lang.RU, text)
            if is_MSVO(doc.get_sentences()[0]):
                print(text, output)
            else:
                print("No", output)
            # words = [(word.text, word.tag, word.dep, word.head.text) for word in doc.get_words()]
            # print(words)
            # words = [word.head.head.text + " " + word.text + " " + word.head.text
            #          for word in doc.get_words() 
            #          if is_attr_clause(word)]
            # words = [" ".join(x.text for x in bfs(word))
            #          for word in doc.get_words() 
            #          if is_part_noun(word)]
            # print(words, output)
