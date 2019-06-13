import os

from rb.core.lang import Lang
from rb.similarity.vector_model import VectorModel, VectorModelType
from rb.utils.downloader import download_model, check_version


class LSA(VectorModel):

    def __init__(self, name: str, lang: Lang, dim: int = 300):
        VectorModel.__init__(self, VectorModelType.LSA, name, lang, dim)  
        