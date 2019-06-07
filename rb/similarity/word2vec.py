import os

from gensim.models import KeyedVectors

from rb.core.lang import Lang
from rb.similarity.vector_model import VectorModel, VectorModelType
from rb.utils.downloader import download_model, check_version


class Word2Vec(VectorModel):

    def __init__(self, name: str, lang: Lang, dim: int = 300):
        VectorModel.__init__(self, VectorModelType.WORD2VEC, name, lang, dim)  
        corpus = "resources/{}/models/{}".format(lang.value, name)
        if check_version(lang, name):
            if not download_model(lang, name):
                raise FileNotFoundError("Requested model ({}) not found for {}".format(name, lang.value))
        model = KeyedVectors.load_word2vec_format("{}/word2vec-{}.txt".format(corpus, dim), binary=False)
        self.vectors = {word: model[word] for idx, word in enumerate(model.index2word)}
