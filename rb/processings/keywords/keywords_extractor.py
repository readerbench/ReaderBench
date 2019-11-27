from rb.similarity.vector_model import VectorModel, VectorModelType
from rb.similarity.vector_model_factory import create_vector_model
from rb.core.document import Document
from rb.core.lang import Lang
from rb.core.word import Word
from collections import Counter
from typing import List, Tuple, Dict
import math
from rb.utils.rblogger import Logger

logger = Logger.get_logger()

class KeywordExtractor():

    vector_model_ro = None
    vector_model_en = None

    def __init__(self):
        pass

    def get_vector_model(self, lang=Lang.RO):
        if lang is Lang.RO:
            if KeywordExtractor.vector_model_ro is None:
                KeywordExtractor.vector_model_ro = create_vector_model(Lang.RO, VectorModelType.from_str('word2vec'), "readme")
            vector_model = KeywordExtractor.vector_model_ro
        elif lang is Lang.EN:
            if KeywordExtractor.vector_model_en is None:
                KeywordExtractor.vector_model_en = create_vector_model(Lang.EN, VectorModelType.from_str("word2vec"), "coca")
            vector_model = KeywordExtractor.vector_model_en
        else:
            logger.error(f'Language {lang.value} is not supported')
            vector_model = None
        return vector_model

    def extract_keywords(self, text, lang=Lang.RO, max_keywords=40) -> List[Tuple[float, Word]]:
        
        logger.info('Loading model for keywords extraction...')
        vector_model = self.get_vector_model(lang=lang)
        logger.info('Computing keywords...')
        doc = Document(lang=lang, text=text)
        lemma_words, raw_words = [], []
        scores = []

        for word in doc.get_words():
            if word.is_content_word() and not word.is_stop:
                lemma_words.append(word.lemma)
                raw_words.append(word)

        counter = Counter(lemma_words)
        taken_words = set()

        # TODO maybe take the max similarity between all words forms - not so impactful
        for ww in raw_words:
            lemma = ww.lemma
            freq = 1 + math.log(counter[lemma])
            sim = vector_model.similarity(ww, doc)
            if lemma not in taken_words:
                taken_words.add(lemma)
                scores.append((sim * freq, ww))
        scores = sorted(scores, reverse=True, key=lambda x: x[0])
        return scores[:max_keywords]

    def keywords_heatmap(self, text: str, lang=Lang.RO, max_keywords=40) -> Dict:

        keywords: List[Tuple[float, Word]]= self.extract_keywords(text=text, lang=lang)
        vector_model = self.get_vector_model(lang=lang)
        doc = Document(lang=lang, text=text)
        
        elements, word_scores = {}, {}

        for kw in keywords:
            word_scores[kw[1].lemma] = {}

        for i, sent in enumerate(doc.get_sentences()):
            elements[str(i + 1)] = sent.text
            for kw in keywords:
                word_scores[kw[1].lemma][str(i + 1)] = str(max(vector_model.similarity(kw[1], sent), 0))

        return {
            "data": {
                "elements": elements,
                "heatmap": {
                    "wordScores": 
                        word_scores
                }
            },
            "success": True,
            "errorMsg":""
        }


    def transform_for_visualization(self, keywords: List[Tuple[int, str]], lang) -> Dict:
        
        vector_model = self.get_vector_model(lang=lang)
        edge_list, node_list = [], []

        for i, kw1 in enumerate(keywords):
            for j, kw2 in enumerate(keywords):
                if i != j and vector_model.similarity(kw1[1], kw2[1]) >= 0.3:
                    edge_list.append({
                        "edgeType":"SemanticDistance",
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
                "nodeList": node_list,
                "success": True,
                "errorMsg": ""
            }
        }

