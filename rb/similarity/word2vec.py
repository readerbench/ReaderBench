import os

from gensim.models import KeyedVectors

from rb.core.lang import Lang
from rb.similarity.vector_model import VectorModel, VectorModelType
from rb.similarity.vector import Vector
from rb.utils.downloader import download_model, check_version
from rb.utils.rblogger import Logger

logger = Logger.get_logger()


class Word2Vec(VectorModel):

    def __init__(self, name: str, lang: Lang, dim: int = 300):
        VectorModel.__init__(self, VectorModelType.WORD2VEC, name, lang, dim)  
        

    def load_vectors(self):
        self.load_vectors_from_txt_file("resources/{}/models/{}/word2vec-{}.txt".format(self.lang.value, self.corpus, self.size))
        
        