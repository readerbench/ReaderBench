import os

from rb.core.lang import Lang
from rb.similarity.vector_model import VectorModel, VectorModelType
from rb.utils.downloader import download_model, check_version


class LDA(VectorModel):

    def __init__(self, name: str, lang: Lang, dim: int = 300):
        VectorModel.__init__(self, VectorModelType.LDA, name, lang, dim)  
        corpus = "resources/{}/models/{}".format(lang.value, name)
        if check_version(lang, name):
            if not download_model(lang, name):
                raise FileNotFoundError("Requested model ({}) not found for {}".format(name, lang.value))
        self.load_vectors_from_txt_file(corpus + "/LDA.model")