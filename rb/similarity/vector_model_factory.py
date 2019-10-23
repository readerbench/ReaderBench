""" do not move this code in vector_model.py, otherwise you will have circular imports"""
from rb.core.lang import Lang
from rb.core.text_element import TextElement
from rb.core.word import Word
from rb.similarity.lda import LDA
from rb.similarity.lsa import LSA
from rb.similarity.vector import Vector
from rb.similarity.vector_model import (CorporaEnum, VectorModel,
                                        VectorModelType)
from rb.similarity.word2vec import Word2Vec

VECTOR_MODELS = {
    Lang.RO: {
        CorporaEnum.README: {
            VectorModelType.WORD2VEC: Word2Vec
        }
    },
    Lang.EN: {
        CorporaEnum.COCA: {
            VectorModelType.WORD2VEC: Word2Vec,
            VectorModelType.LDA: LDA,
            VectorModelType.LSA: LSA
        }
    }
}

def create_vector_model(lang: Lang, model: VectorModelType, corpus: str, dim: int = 300) -> VectorModel:
    try:
        if model == VectorModelType.LSA:
            return LSA(corpus, lang, dim)
        elif model == VectorModelType.LDA:
            return LDA(corpus, lang, dim)
        elif model == VectorModelType.WORD2VEC:
            return Word2Vec(corpus, lang, dim)
    except:
        pass
    return None
