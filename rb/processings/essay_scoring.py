from rb.core.lang import Lang
from rb.core.document import Document
from rb.complexity.complexity_index import ComplexityIndex, compute_indices
from rb.similarity.word2vec import Word2Vec
from rb.cna.cna_graph import CnaGraph
from typing import Tuple, List
from sklearn.svm import SVR
import pickle
import os
import csv
from rb.utils.rblogger import Logger

logger = Logger.get_logger()

class EssayScoring:


    def __init__(self):
        pass

    def compute_indices(self, csv_file_in: str = 'essays.csv', lang: Lang = Lang.RO, 
            write_file: str='essays_eval.csv') -> List[List]:
        w2v = Word2Vec('readme', Lang.RO)        
        all_rows = []
        first_row = ['grade', 'content']
        essay_r = csv.reader(open(csv_file_in, 'rt', encoding='utf-8'))

        for i, row in enumerate(essay_r):
            logger.info('Computing indices for document {}'.format(i))
            try:
                grade = float(row[0])
                content = row[1]
            except:
                print('skip')
                continue
            grade = max(grade, 7.0) - 7.0

            docs_ro = Document(Lang.RO, content)
            CnaGraph(docs_ro, w2v)
            compute_indices(docs_ro)

            row = []
            if len(first_row) == 2:
                for key, v in docs_ro.indices.items():
                    first_row.append(key)
                all_rows.append(first_row)

            row = [grade, content]
            for key, v in docs_ro.indices.items():
                row.append(v)
            
            if len(row) == len(first_row):
                all_rows.append(row)

        with open(write_file, 'wt', encoding='utf-8') as csv_essays_eval:
            csv_writer = csv.writer(csv_essays_eval)
            csv_writer.writerows(all_rows)
        return all_rows
    
    def read_indices(self, path_to_csv_file='essays_eval.csv') -> List[List[float]]:
        essay_r = csv.reader(open(path_to_csv_file, 'rt', encoding='utf-8'))
        results = []
        for i, row in enumerate(essay_r):
            if i == 0:  continue
            results.append([row[0]] + row[2:])
        return results

    def train_svr(self, results: List[List], save_model_file=None):
        total = len(results)
        train_samples = int(total * 0.80)

        train = results[:train_samples]
        test = results[train_samples:]

        y = [float(r[0]) for r in train]
        X = [r[1:] for r in train]

        clf = SVR(gamma='scale', C=1.0, epsilon=0.2)
        clf.fit(X, y)
        if save_model_file:
            pickle.dump(clf, open(save_model_file, 'wb'))

        loss = 0
        grades, dev_in = [], []

        for sample_x in test:
            grades.append(float(sample_x[0]))
            X = sample_x[1:]
            dev_in.append(X)
        res = clf.predict(dev_in)

        for r, grade in zip(res, grades):            
            loss += abs(r - grade)
        loss /= len(res)

        print('loss: {}'.format(loss))

    def predict(self, content, file_to_svr_model, lang=Lang.RO) -> float:
        svr = pickle.load(open(file_to_svr_model, "rb"))
        docs_ro = Document(Lang.RO, content)
        compute_indices(docs_ro)
        indices = []
        for key, v in docs_ro.indices.items():
            indices.append(v)
        grade = svr.predict([indices])[0]
        print(grade)
        return grade

        