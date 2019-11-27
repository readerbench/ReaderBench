from rb.core.lang import Lang
from rb.core.document import Document
from rb.complexity.complexity_index import ComplexityIndex, compute_indices
from rb.similarity.word2vec import Word2Vec
from rb.similarity.vector_model import VectorModelType, CorporaEnum, VectorModel
from rb.similarity.vector_model_factory import VECTOR_MODELS
from typing import Tuple, List
from sklearn.svm import SVR
from sklearn import svm
import pickle
import os
import csv
import random
from werkzeug import secure_filename
import uuid
from rb.cna.cna_graph import CnaGraph
from rb.utils.rblogger import Logger

logger = Logger.get_logger()


class TextClassifier:

    vector_model_ro = None
    vector_model_en = None
    def __init__(self):
        pass
    
    
    def read_indices(self, base_folder: str = 'categories_readme', lang=Lang.RO) -> List[List[float]]:

        categroies = ['general_stats.csv', 'literature_stats.csv', 'science_stats.csv']
        results = []
        indices = []
        if lang is Lang.RO:
            with open('rb/processings/text_classifier/indices_ro_class.txt', 'rt', encoding='utf-8') as f:
                for line in f:
                    indices.append(line.strip())

        for j, cat in enumerate(categroies):
            essay_r = csv.reader(open(os.path.join(base_folder, cat), 'rt', encoding='utf-8'))
            """ first col is the score """
            for i, row in enumerate(essay_r):
                if i == 0:
                    indices_row = row
                    continue
                
                res = [j]
                for findex in indices:
                    for k, rr in enumerate(row):
                        if indices_row[k].strip() == findex:
                            res.append(rr)
                            break
                results.append(res)
        return results

    def train_svm(self, results: List[List], save_model_file=None):
        total = len(results)
        random.shuffle(results)
        train_samples = int(total * 0.80)

        train = results[:train_samples]
        test = results[train_samples:]

        y = [int(r[0]) for r in train]
        print(y)
        X = [r[1:] for r in train]

        clf = svm.SVC(kernel='poly')
        clf.fit(X, y)
        if save_model_file:
            pickle.dump(clf, open(save_model_file, 'wb'))

        classs, dev_in = [], []

        for sample_x in test:
            classs.append(int(sample_x[0]))
            X = sample_x[1:]
            dev_in.append(X)
        res = clf.predict(dev_in)
        right, wrong = 0, 0
        for r, clss in zip(res, classs):
            if r != clss:
                wrong += 1
            else:
                right += 1            

        logger.info('Acc for classification : {}'.format(right/(wrong + right)))

    def predict(self, content: str, file_to_svr_model: str, lang=Lang.RO) -> int:

        svr = pickle.load(open(file_to_svr_model, "rb"))

        doc = Document(lang=lang, text=content)
        if lang is Lang.RO:
            if TextClassifier.vector_model_ro is None:
                TextClassifier.vector_model_ro =  VECTOR_MODELS[lang][CorporaEnum.README][VectorModelType.WORD2VEC](
                    name=CorporaEnum.README.value, lang=lang)
            vector_model = TextClassifier.vector_model_ro
        elif lang is Lang.EN:
            if TextClassifier.vector_model_en is None:
                TextClassifier.vector_model_en = VECTOR_MODELS[lang][CorporaEnum.COCA][VectorModelType.WORD2VEC](
                    name=CorporaEnum.COCA.value, lang=lang)
            vector_model = TextClassifier.vector_model_en

        cna_graph = CnaGraph(doc=doc, models=[vector_model])
        compute_indices(doc=doc, cna_graph=cna_graph)

        indices = []
        if lang is Lang.RO:
            with open('rb/processings/text_classifier/indices_ro_class.txt', 'rt', encoding='utf-8') as f:
                for line in f:
                    indices.append(line.strip())
        values_indices = []
        for ind in indices: 
            for key, v in doc.indices.items():
                if repr(key) == ind:
                    values_indices.append(v)
                    break

        class_txt = svr.predict([values_indices])[0]
        return class_txt

        