import os
from rb.similarity.word_vector_model import WordVectorModel

from rb.core.lang import Lang
from rb.similarity.vector_model import VectorModelType


class LDA(WordVectorModel):

    def __init__(self, name: str, lang: Lang, dim: int = 300, check_updates=True):
        WordVectorModel.__init__(self, VectorModelType.LDA, name, lang, dim, check_updates=check_updates)  

    