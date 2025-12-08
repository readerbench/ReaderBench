from rb.core.lang import Lang
from rb.core.document import Document
from rb.complexity.complexity_index import ComplexityIndex, compute_indices
from rb.similarity.word2vec import Word2Vec
from rb.similarity.vector_model import VectorModelType, CorporaEnum, VectorModel
from rb.similarity.vector_model_factory import VECTOR_MODELS, create_vector_model
from typing import Tuple, List
from sklearn.svm import SVR
from collections import Counter
from sklearn import svm
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
import matplotlib.pyplot as plt
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


    def __init__(self):
        pass
    
    def get_vector_model(self, lang: Lang = Lang.RO) -> VectorModel:
        global logger
        if lang is Lang.RO:
            vector_model = create_vector_model(Lang.RO, VectorModelType.from_str('word2vec'), "readme")
        elif lang is Lang.EN:
            vector_model = create_vector_model(Lang.EN, VectorModelType.from_str("word2vec"), "coca")
        else:
            logger.error(f'Language {lang.value} is not supported for essay scoring task')
            vector_model = None
        return vector_model
    
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
        train_samples = int(total * 0.8)

        train = results[:train_samples]
        test = results[train_samples:]

        y = [int(r[0]) for r in train]
        X = [r[1:] for r in train]

        clf = svm.SVC(kernel='poly', degree=14, class_weight={0: 0.1, 1: 0.6, 2: 0.3}).fit(X, y)
        if save_model_file:
            pickle.dump(clf, open(save_model_file, 'wb'))

        dev_out, dev_in = [], []

        for sample_x in test:
            if int(sample_x[0]) == 0 and random.random() < 0.7:
                continue
            dev_out.append(int(sample_x[0]))
            Xx = sample_x[1:]
            dev_in.append(Xx)
        print(Counter(dev_out))
        disp = plot_confusion_matrix(clf, dev_in, dev_out, display_labels=['general', 'science', 'literature', ])
        
        res = clf.predict(dev_in)
        disp.ax_.set_title('Confusion Matrix')
        right, wrong = 0, 0
        for r, clss in zip(res, dev_out):
            if r != clss:
                wrong += 1
            else:
                right += 1            

        logger.info('Acc for classification : {}'.format(right/(wrong + right)))
        plt.show()

    def predict(self, content: str, file_to_svr_model: str, lang=Lang.RO) -> int:

        svr = pickle.load(open(file_to_svr_model, "rb"))

        doc = Document(lang=lang, text=content)
        vector_model = self.get_vector_model(lang=lang)

        cna_graph = CnaGraph(docs=doc, models=[vector_model])
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

        