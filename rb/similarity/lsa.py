import os
from rb.similarity.word_vector_model import WordVectorModel

from rb.core.lang import Lang
from rb.similarity.vector_model import VectorModelType
from rb.utils.downloader import download_model, check_version


class LSA(WordVectorModel):

    def __init__(self, name: str, lang: Lang, dim: int = 300, check_updates=True):
        WordVectorModel.__init__(self, VectorModelType.LSA, name, lang, dim, check_updates=check_updates)  
        