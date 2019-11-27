from gensim.models.keyedvectors import KeyedVectors
from gensim.models import Word2Vec, LdaMulticore, LsiModel, FastText, KeyedVectors, LdaModel
from gensim.test.utils import datapath, get_tmpfile
from gensim.corpora import Dictionary, MmCorpus
import gensim
import pyLDAvis
import pyLDAvis.gensim
from rb.utils.utils import split_sentences, tokenize_docs, load_docs
from typing import Dict, List, Iterable
from os.path import dirname
import numpy as np
from itertools import islice
from rb.parser.spacy_parser import SpacyParser
from math import log
import time
import itertools
from rb.core.lang import Lang
from rb.utils.rblogger import Logger
import logging
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
from os import listdir
from os.path import isdir, isfile, join

logger = Logger.get_logger()
# use this to disable loggins for gensim library
logging.getLogger("gensim").setLevel(logging.WARNING)

class Preprocess(object):
    def __init__(self, parser: SpacyParser, folder: str, 
                lang: Lang, split_sent: bool = True, only_dict_words: bool = False) -> None:
        if only_dict_words:
            self.test = lambda x: not x.is_oov
        else:
            self.test = lambda x: True
        self.lang = lang
        self.folder = folder
        self.split_sent = split_sent
        self.only_dict_words = only_dict_words
        self.parser = parser

    def __iter__(self):
        for doc in load_docs(self.folder):
            #print('processing doc {}'.format(doc))
            result = []
            doc = self.parser.preprocess(doc, self.lang)
            for tokens in self.parser.get_tokens_lemmas(self.parser.tokenize_sentences(doc), self.lang):
                if len(tokens) == 0:
                    continue
                tokens = [token.lemma_.lower() for token in tokens if not token.is_stop and token.is_alpha and self.test(token)]
                if self.split_sent:
                    yield tokens
                else:
                    result += tokens
            if not self.split_sent:
                yield result

def train_w2v(sentences: Preprocess, outputFolder: str):
    global logger
    logger.info("Starting training word2vec...")
    model = Word2Vec(size=300, window=5, min_count=5, workers=16)
    model.build_vocab(sentences=sentences)
    total_words = model.corpus_total_words  # number of words in the corpus
    total_examples = model.corpus_count # examples aka sentences
    logger.info("Words in vocabulary: {}, examples: {}".format(total_words, total_examples))

    model.train(sentences=sentences, total_words=total_words, total_examples=total_examples, epochs=5)  # train
    path = outputFolder + "/word2vec.model"
    model.save(path)
    logger.info("Model word2vec saved to {}".format(path))

# load_word2vec_format should be True if the model was saved with saved  model.wv.save_word2vec_format(path, binary=False)
# if the model was saved simply with model.save(path) load_word2vec_format should be False
def test_load_w2v(path_w2vec: str, load_word2vec_format=False):
    test_words = ["alunec", 'merg', 'ajung', 'plec', 'coleg']
    if load_word2vec_format == True:
        model = KeyedVectors.load_word2vec_format(path_w2vec, binary=False)
    else:
        model = Word2Vec.load(path_w2vec)
    for tw in test_words:
        if tw in model.wv:
            logger.info('Word vector for {} {}'.format(tw, model.wv[tw]))
    logger.info('word2vec loaded successfully')

def train_fast_text(sentences: Preprocess, outputFolder: str):
    logger.info("Starting training fast text ...")
    model = FastText(size=300, window=5, min_count=5, workers=16)
    model.build_vocab(sentences=sentences)
    total_words = model.corpus_total_words  # number of words in the corpus
    total_examples = model.corpus_count # examples aka sentences
    logger.info("Words in vocabulary: {}, examples: {}".format(total_words, total_examples))

    model.train(sentences=sentences, total_words=total_words, total_examples=total_examples, epochs=5)  # train
    path = outputFolder + "/fast_text.model"
    model.save(path)
    logger.info("Model fast text saved to {}".format(path))

def test_load_fast_text(path_fast_text: str):
    fname = get_tmpfile(path_fast_text)
    test_word = "istorie"
    model = FastText.load(fname)
    logger.info('Word vector for {} {}'.format(test_word, model.wv[test_word]))

def train_lda(docs: List, outputFolder: str):
    docs = list(docs)
    id2word = Dictionary(docs)
    id2word.filter_extremes(no_below=8, no_above=0.1, keep_n=1000000)
    logger.info('Starting trainin lda model with {} docs {} tokens'.format(id2word.num_docs, id2word.num_pos))
    corpus = [id2word.doc2bow(doc) for doc in docs]
    # saved for visualuzations
    MmCorpus.serialize(join(outputFolder, 'lda_corpus.mm'), corpus)
    logger.info('Corpus saved.')
    logger.info("Starting training lda...")
    lda = LdaMulticore(corpus, num_topics=300, id2word=id2word)
    path = join(outputFolder, "lda.model")

    #matrix = np.transpose(lda.get_topics())
    # with open(path, "wt", encoding='utf-8') as f:
    #     f.write("{} {}\n".format(np.size(matrix, 0), np.size(matrix, 1)))
    #     for idx in range(np.size(matrix, 0)):
    #         f.write(id2word[idx] + " " + " ".join([str(x) for x in matrix[idx]]) + "\n")
    lda.save(path)
    logger.info("Model lda saved to {}".format(path))

def test_load_lda(path_to_model: str):
    model = LdaModel.load(path_to_model)
    for i in range(10):
        logger.info(model.show_topic(i, 30))
    logger.info('LDA loaded successfully')

def visualize_lda(path_to_lda: str, path_to_corpus_lda: str):
    logger.info('Loadin lda model ...')
    corpus = gensim.corpora.MmCorpus(path_to_corpus_lda)
    ldaModel = gensim.models.LdaModel.load(path_to_lda)
    id2word = ldaModel.id2word
    p = pyLDAvis.gensim.prepare(ldaModel, corpus, id2word,  mds='mmds')
    pyLDAvis.save_html(p, 'lda.html')
    logger.info('LDA visualization saved in lda.html file')

def log_entropy_norm(corpus: List) -> List:
    result = []
    for doc in corpus:
        if len(doc) == 0:
            result.append(doc)
            continue
        total = sum(count for idx, count in doc)
        entropy = sum((count / total) * log(count / total) for idx, count in doc)
        entropy = 1 + (entropy / log(len(corpus)))
        result.append([(idx, log(1 + count) * entropy) for idx, count in doc])
    return result


def train_lsa(docs: Iterable, outputFolder: str):
    docs = list(docs)
    id2word = Dictionary(docs)
    id2word.filter_extremes(no_below=7, no_above=0.1, keep_n=1000000)
    corpus = [id2word.doc2bow(doc) for doc in docs]
    corpus = log_entropy_norm(corpus)
    logger.info('Lsa, docs: {}, dictionary: {}'.format(len(docs), len(id2word)))
    logger.info("Starting training lsa...")
    
    lsa = LsiModel(corpus=corpus, id2word=id2word, num_topics=300)
    path = outputFolder + "/lsa.model"
    lsa.save(outputFolder + "/lsa.bin")
    matrix = np.transpose(lsa.get_topics())
    # lsa.get_topics() shape is  (num_topics, vocabulary_size)
    # transpose to use it as word embeddings
    with open(path, "wt", encoding='utf-8') as f:
        f.write("{} {}\n".format(np.size(matrix, 0), np.size(matrix, 1)))
        for idx in range(np.size(matrix, 0)):
            f.write(id2word[idx] + " " + " ".join([str(x) for x in matrix[idx]]) + "\n")
    logger.info("Model lsa saved to {}".format(path))

# TODO, model was saved manually, load manually
def test_load_lsa(path_lsa: str):
    model = LsiModel.load(path_lsa)
    logger.info(model.projection.s)
    logger.info('LSA loaded successfully')

# parse tokens, get lemmas, only lowercase letters
def preprocess(parser: SpacyParser, folder: str, lang: Lang,
     split_sent: bool = True, only_dict_words: bool = False) -> Iterable[List]:
    if only_dict_words:
        test = lambda x: not x.is_oov
    else:
        test = lambda x: True
    for doc in load_docs(folder):
        #print('processing doc {}'.format(doc))
        result = []
        doc = parser.preprocess(doc, lang)
        for tokens in parser.get_tokens_lemmas(parser.tokenize_sentences(doc), lang):
            if len(tokens) == 0:
                continue
            tokens = [token.lemma_.lower() for token in tokens if not token.is_stop and token.is_alpha and test(token)]
            if split_sent:
                yield tokens
            else:
                result += tokens
        if not split_sent:
            yield result

if __name__ == "__main__":
    #inputFolder= "resources/corpora/FR/Le Monde"
    inputFolder = "/home/teo/projects/newsanalyser/dumped-news-sample"
    parser = SpacyParser.get_instance()
    logger.info("Loading dataset from {}".format(inputFolder))
    sentences = Preprocess(parser, inputFolder, Lang.RO, split_sent=False, only_dict_words=False)
    # #train_fast_text(sentences, inputFolder)
    #train_lsa(sentences, inputFolder)
    #train_w2v(sentences, inputFolder)
    train_lda(sentences, inputFolder)
    
    visualize_lda(path_to_lda=join(inputFolder, "lda.model"),
                  path_to_corpus_lda=join(inputFolder, "lda_corpus.mm"))

    #test_load_w2v("/home/teo/projects/news-cosmin/models/word2vec.model", load_word2vec_format=False)
    #test_load_fast_text("/home/teo/projects/readme-models/models/fasttext2/fast_text.model")
    
    #test_load_lsa("/home/teo/projects/readme-models/models/lsa/lsa.bin")
    #test_load_w2v("/home/teo/projects/readme-models/models/word2vec.model", True)

    # model = LsiModel.load(inputFolder + "/lsa.bin")
    # print(model.projection.s)
#from gensim.models.keyedvectors import KeyedVector
