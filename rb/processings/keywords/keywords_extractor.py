from rb.similarity.vector_model import VectorModel, VectorModelType
from rb.similarity.vector_model_factory import create_vector_model
from rb.core.document import Document
from rb.core.lang import Lang
from rb.core.word import Word
from rb.core.text_element_type import TextElementType
from collections import Counter
from typing import List, Tuple, Dict, Set
import math
from rb.utils.rblogger import Logger

logger = Logger.get_logger()


class KeywordExtractor():

    def __init__(self):
        pass

    def get_vector_model(self, lang: Lang = Lang.RO) -> VectorModel:
        global logger
        if lang is Lang.RO:
            vector_model = create_vector_model(
                Lang.RO, VectorModelType.from_str('word2vec'), "readme")
        elif lang is Lang.EN:
            vector_model = create_vector_model(
                Lang.EN, VectorModelType.from_str("word2vec"), "coca")
        elif lang is Lang.FR:
            vector_model = create_vector_model(
                Lang.FR, VectorModelType.from_str("word2vec"), "le_monde_small")                
        elif lang is Lang.ES:
            vector_model = create_vector_model(
                Lang.ES, VectorModelType.from_str("word2vec"), "jose_antonio") 
        elif lang is Lang.RU:
            vector_model = create_vector_model(
                Lang.RU, VectorModelType.from_str("word2vec"), "rnc_wikipedia") 
        else:
            logger.error(
                f'Language {lang.value} is not supported for keywords task')
            vector_model = None
        return vector_model

    def extract_keywords(self, text: str, lang: Lang = Lang.RO, max_keywords: int = 40, vector_model: VectorModel = None) -> List[Tuple[float, Word]]:

        if vector_model is None:
            vector_model = self.get_vector_model(lang=lang)
        logger.info('Computing keywords...')
        doc: Document = Document(lang=lang, text=text)
        lemma_words, raw_words = [], []
        scores: List[Tuple[float, Word]] = []

        for word in doc.get_words():
            if word.is_content_word() and not word.is_stop:
                lemma_words.append(word.lemma)
                raw_words.append(word)

        counter = Counter(lemma_words)
        taken_words: Set[str] = set()

        # TODO maybe take the max similarity between all words forms - not so impactful though - Teo
        for ww in raw_words:
            lemma = ww.lemma
            freq = 1 + math.log(counter[lemma])
            sim = vector_model.similarity(ww, doc)
            if lemma not in taken_words:
                taken_words.add(lemma)
                scores.append((sim * freq, ww))

        scores = sorted(scores, reverse=True, key=lambda x: x[0])
        return scores[:max_keywords]

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
