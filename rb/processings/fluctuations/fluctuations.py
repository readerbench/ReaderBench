from rb.core.lang import Lang
from rb.core.document import Document
from rb.complexity.complexity_index import ComplexityIndex, compute_indices
from rb.complexity.cohesion.adj_cohesion import AdjCohesion
from rb.similarity.word2vec import Word2Vec
from rb.similarity.vector_model import VectorModelType, CorporaEnum, VectorModel
from rb.similarity.vector_model_factory import VECTOR_MODELS, create_vector_model
from rb.cna.cna_graph import CnaGraph
from typing import Tuple, List
from sklearn.svm import SVR
import pickle
import os
import csv
from rb.utils.rblogger import Logger
from typing import List, Tuple
import numpy as np


logger = Logger.get_logger()


class Fluctuations:

    def __init__(self):
        pass
    
    def get_vector_model(self, lang: Lang = Lang.RO) -> VectorModel:
        global logger
        if lang is Lang.RO:
            vector_model = create_vector_model(Lang.RO, VectorModelType.from_str('word2vec'), "readme")
        elif lang is Lang.EN:
            vector_model = create_vector_model(Lang.EN, VectorModelType.from_str("word2vec"), "coca")
        else:
            logger.error(f'Language {lang.value} is not supported for fluctuations task')
            vector_model = None
        return vector_model

    def compute_thresholds(self, values: List[float]) -> Tuple[int, int]:
        if len(values) > 1:
            stdev = np.std(values)
            avg = np.mean(values)
        elif len(values) == 1:
            avg = values[0]
            stdev = 1
        else:
            avg = -1
            stdev = -1
        return (max(0, avg + 2.0 * stdev), max(0, avg - 2.0 * stdev))


    def compute_indices(self, text: str, lang: Lang) -> List[List]:
        doc = Document(lang=lang, text=text)
        vector_model = self.get_vector_model(lang=lang)
        cna_graph = CnaGraph(docs=doc, models=[vector_model])
        compute_indices(doc=doc, cna_graph=cna_graph)

        indices_sent = {
                        'AvgSentUnqPOSMain_noun': 
                            {Lang.RO: 'Numărul de substantive unice per propoziție',
                             Lang.EN: 'Number of unique nouns per sentence'}, 
                        'AvgSentUnqPOSMain_verb': 
                            {Lang.RO: 'Numărul de verbe unice per propoziție',
                             Lang.EN: 'Number of unique verbs per sentence'},
                        'AvgSentUnqPOSMain_adj':
                            {Lang.RO: 'Numărul de adjective unice per propoziție',
                             Lang.EN: 'Number of unique adjectives per sentence'},
                        'AvgSentUnqPOSMain_adv':
                            {Lang.RO: 'Numărului de adverbe unice per propoziție',
                             Lang.EN: 'Number of unique adverbs per sentence'},
                        'AdjExtCoh_SENT':
                            {Lang.RO: 'Coeziunea propoziției curente cu propozițiile vecine',
                             Lang.EN: 'Cohesion of the current sentence with its neighbouring sentences'}
                        }
        indices_block = {
                        'AvgSentUnqPOSMain_noun': 
                            {Lang.RO: 'Media numărului de substantive unice per frază',
                             Lang.EN: 'Average of the number of unique nouns per paragraph'}, 
                        'AvgSentUnqPOSMain_verb': 
                            {Lang.RO: 'Media numărului de verbe unice per frază',
                             Lang.EN: 'Average of the number of unique verbs per paragraph'},
                        'AvgSentUnqPOSMain_adj':
                            {Lang.RO: 'Media numărului de adjective unice per frază',
                             Lang.EN: 'Average of the number of unique adjectives per paragraph'},
                        'AvgSentUnqPOSMain_adv':
                            {Lang.RO: 'Media numărului de adverbe unice per frază',
                             Lang.EN: 'Average of the number of unique adverbs per paragraph'},
                        'AdjExtCoh_BLOCK':
                            {Lang.RO: 'Coeziunea paragrafului curent cu paragrafele vecine',
                             Lang.EN: 'Cohesion of the current paragraph with its neighbouring paragraphs'}
                        }
        result = []

        for ind_sent, _ in indices_sent.items():
            d = {'index': ind_sent, 'index_description': indices_sent[ind_sent][lang],
                 'level': 'sentence', 'values': [], 'text': []}
            for sent in doc.get_sentences():
                for key, v in sent.indices.items():
                    if repr(key) == ind_sent:
                        d['values'].append(v)
                        d['text'].append(sent.text)
            maxt, mint = self.compute_thresholds(d['values'])
            d['threshold'] = {
                'min': str(mint),
                'max': str(maxt)
            }
            for i, v in enumerate(d['values']):
                d['values'][i] = str(v)
            result.append(d)
        
        for ind_block, _ in indices_block.items():
            d = {'index': ind_block, 'index_description': indices_block[ind_block][lang],
                 'level': 'paragraph', 'values': [], 'text': []}
            for block in doc.get_blocks():
                for key, v in block.indices.items():
                    if repr(key) == ind_block:
                        d['values'].append(v)
                        d['text'].append(block.text)
            maxt, mint = self.compute_thresholds(d['values'])
            d['threshold'] = {
                'min': str(mint),
                'max': str(maxt)
            }

            for i, v in enumerate(d['values']):
                d['values'][i] = str(v)
            result.append(d)
        return result