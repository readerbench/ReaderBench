""" do not move this code in vector_model.py, otherwise you will have circular imports"""
from rb.similarity.word2vec import Word2Vec
from rb.similarity.lda import LDA
from rb.similarity.lsa import LSA
from rb.core.lang import Lang
from rb.core.text_element import TextElement
from rb.core.word import Word
from rb.similarity.vector import Vector
from rb.similarity.vector_model import CorporaEnum, VectorModel, VectorModelType

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