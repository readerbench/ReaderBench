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


def extract_keywords(self, text: Union[str, TextElement], lang: Lang = Lang.RO, max_keywords: int = 40, vector_model: VectorModel = None, threshold: float = 0.3) -> List[Tuple[float, str]]:

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

def keywords_heatmap(self, text: str, lang: Lang = Lang.RO, granularity: TextElementType = TextElementType.SENT,
                        max_keywords: int = 40) -> Dict:

    logger.info('Loading model for keywords extraction...')
    keywords: List[Tuple[float, Word]] = self.extract_keywords(
        text=text, lang=lang)
    vector_model: VectorModel = self.get_vector_model(lang=lang)
    doc: Document = Document(lang=lang, text=text)
    logger.info('Computing keywords heatmap...')

    elements, word_scores = {}, {}

    for kw in keywords:
        word_scores[kw[1].lemma] = {}

    if granularity is TextElementType.SENT:
        for i, sent in enumerate(doc.get_sentences()):
            elements[str(i + 1)] = sent.text
            for kw in keywords:
                word_scores[kw[1].lemma][str(
                    i + 1)] = str(max(vector_model.similarity(kw[1], sent), 0))
    else:
        for i, block in enumerate(doc.get_blocks()):
            elements[str(i + 1)] = block.text
            for kw in keywords:
                word_scores[kw[1].lemma][str(
                    i + 1)] = str(max(vector_model.similarity(kw[1], block), 0))

    return {
        "data": {
            "elements": elements,
            "heatmap": {
                "wordScores":
                    word_scores
            }
        },
        "success": True,
        "errorMsg": ""
    }

def transform_for_visualization(self, keywords: List[Tuple[int, Word]], lang: Lang) -> Dict:

    vector_model: VectorModel = self.get_vector_model(lang=lang)
    edge_list, node_list = [], []

    for i, kw1 in enumerate(keywords):
        for j, kw2 in enumerate(keywords):
            if i != j and vector_model.similarity(kw1[1], kw2[1]) >= 0.3:
                edge_list.append({
                    "edgeType": "SemanticDistance",
                    "score": str(max(vector_model.similarity(kw1[1], kw2[1]), 0)),
                    "sourceUri": kw1[1].lemma,
                    "targetUri": kw2[1].lemma
                })

    for kw in keywords:
        node_list.append({
            "type": "Word",
            "uri": kw[1].lemma,
            "displayName": kw[1].lemma,
            "active": True,
            "degree": str(max(0, float(kw[0])))
        })

    return {
        "data": {
            "edgeList": edge_list,
            "nodeList": node_list
        },
        "success": True,
        "errorMsg": ""
    }
