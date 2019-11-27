import os

from gensim.models import KeyedVectors

from rb.core.lang import Lang
from rb.similarity.vector_model import VectorModel, VectorModelType
from rb.similarity.vector import Vector
from rb.utils.downloader import download_model, check_version


class Word2Vec(VectorModel):

    def __init__(self, name: str, lang: Lang, dim: int = 300):
        VectorModel.__init__(self, VectorModelType.WORD2VEC, name, lang, dim)  
        

    def load_vectors(self):
        model = KeyedVectors.load_word2vec_format("resources/{}/models/{}/word2vec-{}.txt".format(self.lang.value, self.corpus, self.size), binary=False)
        self.vectors = {word: Vector(model[word]) for idx, word in enumerate(model.index2word)}
        