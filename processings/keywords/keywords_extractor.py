import bisect
import math
from collections import Counter
from typing import Dict, List, Set, Tuple, Union

from rb.core.document import Document
from rb.core.lang import Lang
from rb.core.text_element_type import TextElementType
from rb.core.word import Word
from rb.similarity.vector_model import VectorModel, VectorModelType
from rb.similarity.vector_model_factory import create_vector_model, get_default_model
from rb.utils.rblogger import Logger
from rb.core.text_element import TextElement

logger = Logger.get_logger()


def extract_keywords(text: Union[str, TextElement], lang: Lang = Lang.RO, max_keywords: int = 40, vector_model: VectorModel = None, threshold: float = 0.3) -> List[Tuple[float, str]]:

    if vector_model is None:
        vector_model = get_default_model(lang)
    logger.info('Computing keywords...')
    if not isinstance(text, TextElement):
        doc = Document(lang=lang, text=text)
    else:
        doc = text
    lemma_words, raw_words = [], []
    scores: List[Tuple[float, Word]] = []

    for word in doc.get_words():
        if word.is_content_word() and not word.is_stop:
            raw_words.append(word)

    counter = Counter(raw_words)
    
    score_dict = {}
    for word, freq in counter.items():
        if word.lemma not in score_dict:
            score_dict[word.lemma] = 0
        freq = 1 + math.log(freq)
        sim = vector_model.similarity(word, doc)
        score_dict[word.lemma] += sim * freq

    scores = [(score, lemma) for lemma, score in score_dict.items()]
    scores = sorted(scores, reverse=True, key=lambda x: x[0])
    scores.reverse()
    idx = bisect.bisect(scores, (threshold, ))
    scores.reverse()
    if max_keywords == -1:
        return scores
    return scores[:len(scores)-idx]
    # return scores[:max_keywords]

