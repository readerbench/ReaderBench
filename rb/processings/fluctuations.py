from rb.core.lang import Lang
from rb.core.document import Document
from rb.complexity.complexity_index import ComplexityIndex, compute_indices
from rb.complexity.cohesion.adj_cohesion import AdjCohesion
from rb.similarity.word2vec import Word2Vec
from rb.cna.cna_graph import CnaGraph
from typing import Tuple, List
from sklearn.svm import SVR
import pickle
import os
import csv
from rb.utils.rblogger import Logger
from typing import List, Tuple
import numpy as np

log = open('log.log', 'wt', encoding='utf-8')

class Fluctuations:

    def __init__(self):
        pass
    
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
        return (max(0, avg + 2.5 * stdev), max(0, avg - 2.5 * stdev))


    def compute_indices(self, text: str, lang: Lang) -> List[List]:
        
        if lang is Lang.RO:
            w2v = Word2Vec('readme', lang)
        else:
            w2v = Word2Vec('coca', lang)

        doc = Document(lang, text)
        CnaGraph(doc, w2v)
        compute_indices(doc)

        indices_sent = ['AvgWordUnqPOSMain_noun', 'AvgWordUnqPOSMain_verb', 'AvgWordUnqPOSMain_adj',
                        'AvgWordUnqPOSMain_adv', 'AvgWordUnqPOSMain_pron', 'AdjExtCoh_SENT']
        indices_block = ['AvgSentUnqPOSMain_noun', 'AvgSentUnqPOSMain_verb', 'AvgSentUnqPOSMain_adj',
                         'AvgSentUnqPOSMain_adv', 'AvgSentUnqPOSMain_pron', 'AdjExtCoh_BLOCK']
        result = []

        for ind_sent in indices_sent:
            d = {'index': ind_sent, 'level': 'sentence', 'values': [], 'text': []}
            for sent in doc.get_sentences():
                for key, v in sent.indices.items():
                    if str(key) == ind_sent:
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
        
        for ind_block in indices_block:
            d = {'index': ind_block, 'level': 'sentence', 'values': [], 'text': []}
            for block in doc.get_blocks():
                for key, v in block.indices.items():
                    if str(key) == ind_block:
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