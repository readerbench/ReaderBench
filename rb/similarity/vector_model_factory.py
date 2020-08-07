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
from rb.utils.rblogger import Logger

logger = Logger.get_logger() 


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
    },
    Lang.ES: {
        CorporaEnum.JOSE_ANTONIO: {
            VectorModelType.WORD2VEC: Word2Vec
        }
    },
    Lang.RU: {
        CorporaEnum.RNC_WIKIPEDIA: {
            VectorModelType.WORD2VEC: Word2Vec
        }
    }
}

DEFAULT_MODELS = {
    Lang.RO: (VectorModelType.WORD2VEC, CorporaEnum.README),
    Lang.EN: (VectorModelType.WORD2VEC, CorporaEnum.COCA),
    Lang.ES: (VectorModelType.WORD2VEC, CorporaEnum.JOSE_ANTONIO),
    Lang.RU: (VectorModelType.WORD2VEC, CorporaEnum.RNC_WIKIPEDIA),    
}

EXISTING_MODELS = {}


def save_model(vector_model: VectorModel, lang: Lang, model_type: VectorModelType, 
                corpus: str, dim: int, model_abbr: str) -> bool:
    global EXISTING_MODELS, logger

    if lang not in EXISTING_MODELS:
        EXISTING_MODELS[lang] = {}
    if model_type not in EXISTING_MODELS[lang]:
        EXISTING_MODELS[lang][model_type] = {}
    if corpus not in EXISTING_MODELS[lang][model_type]:
        EXISTING_MODELS[lang][model_type][corpus] = {}
    if dim not in EXISTING_MODELS[lang][model_type][corpus]:
        logger.info(f'Model {model_abbr} does not exists in memory. Saving it...')
        EXISTING_MODELS[lang][model_type][corpus][dim] = vector_model
        return True
    return False

def get_existing_model(lang: Lang, model: VectorModelType,  corpus: str, dim: int, model_abbr: str) -> VectorModel:
    global EXISTING_MODELS

    if (lang in EXISTING_MODELS and model in EXISTING_MODELS[lang] and corpus in EXISTING_MODELS[lang][model]
        and dim in EXISTING_MODELS[lang][model][corpus]):
        logger.info(f'Model {model_abbr} already exists in memory')
        return EXISTING_MODELS[lang][model][corpus][dim]
    return None

def get_default_model(lang: Lang) -> VectorModel:
    if lang not in DEFAULT_MODELS:
        return None
    model, corpus = DEFAULT_MODELS[lang]
    return create_vector_model(lang, model, corpus=corpus.value)

    
def create_vector_model(lang: Lang, model: VectorModelType, corpus: str, dim: int = 300, check_updates=True) -> VectorModel:
    global logger
    model_abbr = f'{lang.value}-{model.name}-{corpus}-{dim}'
    logger.info(f'Getting model {model_abbr}...')
    vector_model = get_existing_model(lang=lang, model=model, corpus=corpus, dim=dim, model_abbr=model_abbr)
    if vector_model is not None:
        return vector_model
    else:
        try:
            if model == VectorModelType.LSA:
                vector_model = LSA(corpus, lang, dim, check_updates=check_updates)
                save_model(vector_model=vector_model, lang=lang, 
                        model_type=model, corpus=corpus, dim=dim, model_abbr=model_abbr)
            elif model == VectorModelType.LDA:
                vector_model = LDA(corpus, lang, dim, check_updates=check_updates)
                save_model(vector_model=vector_model, lang=lang, 
                        model_type=model, corpus=corpus, dim=dim, model_abbr=model_abbr)
            elif model == VectorModelType.WORD2VEC:
                vector_model = Word2Vec(corpus, lang, dim, check_updates=check_updates)
                save_model(vector_model=vector_model, lang=lang, 
                        model_type=model, corpus=corpus, dim=dim, model_abbr=model_abbr)
            else:
                logger.error(f'Model {model_abbr} could not be instantiate.')
                return None
            return vector_model
        except:
            logger.error(f'Model {model_abbr} could not be instantiate.')
        return None
