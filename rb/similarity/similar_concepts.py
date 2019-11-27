from rb.core.lang import Lang
from rb.similarity.vector_model import VectorModel
import rb.similarity.wordnet as wordnet
from typing import List


def get_similar_concepts(word: str, lang: Lang, semantic_models: List[VectorModel]) -> List[str]:

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

    return similar_concepts