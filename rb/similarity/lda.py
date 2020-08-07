import os

from rb.core.lang import Lang
from rb.similarity.vector_model import VectorModel, VectorModelType


class LDA(VectorModel):

    def __init__(self, name: str, lang: Lang, dim: int = 300, check_updates=True):
        VectorModel.__init__(self, VectorModelType.LDA, name, lang, dim, check_updates=check_updates)  

    