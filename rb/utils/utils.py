from gensim.models import Word2Vec 
import spacy
from typing import List, Iterable
from nltk.tokenize import sent_tokenize, WordPunctTokenizer
from os.path import isdir, isfile, join
from os import listdir

from spacy.lang.ro.lemmatizer import LOOKUP
#from spacy.lang.ro.lex_attrs import words

# nlp = spacy.load('en_core_web_sm')
# nlp.remove_pipe('tagger')
# nlp.remove_pipe('parser')
# nlp.remove_pipe('ner')

def split_sentences(fileName: str) -> Iterable[List[str]]:
    # sentences = []
    tokenizer = WordPunctTokenizer()
    with open(fileName, "rt", encoding='utf-8') as f:
        for line in f.readlines():
            for sent in sent_tokenize(line):
                yield [token for token in tokenizer.tokenize(sent)
                       if token.isalpha and not token == '.']

def tokenize_docs(fileName: str) -> Iterable[List[str]]:
    # sentences = []
    tokenizer = WordPunctTokenizer()
    with open(fileName, "rt", encoding='utf-8') as f:
        for line in f.readlines():
            yield [token for token in tokenizer.tokenize(line)
                if token.isalpha and not token == '.']
            
def load_docs(folder: str) -> Iterable[str]:
    for f in listdir(folder):
        if isfile(join(folder, f)) and f.endswith(".txt"):
            with open(join(folder, f), "r") as fin:
                current = ""
                line = fin.readline()
                while line != "":
                    if line == "\n":
                        yield current
                        current = ""
                    else:
                        current += line
                    line = fin.readline()
                if current != "":
                    yield current

if __name__ == "__main__":
    pass
    # with open("dict_ro.txt", "wt") as out:
    #     all_words = words | {word for word in LOOKUP} | {lemma for word, lemma in LOOKUP.items()}
    #     all_words = sorted(list(all_words))
    #     for word in all_words:
    #         out.write(word + "\n")

