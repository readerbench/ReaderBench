from rb.core.lang import Lang
from rb.similarity.vector_model import VectorModel, VectorModelType
from rb.similarity.vector_model_factory import create_vector_model
from rb.core.word import Word
import rb.similarity.wordnet as wordnet
from typing import List
from typing import Dict

def create_semantic_models(lang: Lang) -> List[VectorModel]:
    if lang is Lang.RO:
        return [create_vector_model(Lang.RO, VectorModelType.from_str('word2vec'), "readme")]
    if lang is Lang.EN:
        semantic_models = []
        semantic_models.append(create_vector_model(Lang.EN, VectorModelType.from_str("word2vec"), "coca"))
        # semantic_models.append(create_vector_model(Lang.EN, VectorModelType.from_str("lsa"), "coca"))
        # semantic_models.append(create_vector_model(Lang.EN, VectorModelType.from_str("lda"), "coca"))
        return semantic_models
    else:
        return None


def avg_similarity_using_models(word1: Word, word2: Word, semantic_models: List[VectorModel]) -> float:
        avg_models_similarity = 0.0
        for sm in semantic_models:
            avg_models_similarity += sm.similarity(word1, word2)
        if semantic_models:
            avg_models_similarity /= len(semantic_models)
        return avg_models_similarity


def dictify_similar_concepts(word: Word, lang: Lang, similar_concepts: List[str], semantic_models: List[VectorModel]) -> Dict:
    data = {}
    concepts = {}
    for concept in similar_concepts:
        concept_word = Word.from_str(lang, concept)
        concepts["{}({}, null, null)".format(concept, concept_word.lemma)] = avg_similarity_using_models(word, concept_word, semantic_models)
    
    data["concepts"] = concepts

    return {"data": data}


def get_similar_concepts(raw_word: str, lang: Lang) -> List[str]:
    semantic_models = create_semantic_models(lang)
    if not semantic_models:
        return []
    
    word = Word.from_str(lang, raw_word)
    synonyms = wordnet.get_synonyms(word)
    hypernyms = wordnet.get_hypernyms(word)
    similar_concepts = []
    similar_concepts.extend(synonyms)
    similar_concepts.extend(hypernyms)

    for vect_model in semantic_models:
        closest_semantic_words = vect_model.most_similar(word, topN=5, threshold=0.5)
    similar_concepts.extend([x[0] for x in closest_semantic_words])
    similar_concepts = list(set(similar_concepts))
    # remove the word if that is the case
    similar_concepts = [x for x in similar_concepts if x != word]

    response = dictify_similar_concepts(word, lang, similar_concepts, semantic_models)
    response["success"] = True
    response["errorMsg"] = ""

    return response