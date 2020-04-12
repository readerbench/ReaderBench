from rb.core.lang import Lang
from rb.core.document import Document
from rb.complexity.complexity_index import ComplexityIndex, compute_indices
from rb.similarity.word2vec import Word2Vec
from rb.similarity.vector_model import VectorModelType, CorporaEnum, VectorModel
from rb.similarity.vector_model_factory import VECTOR_MODELS, create_vector_model
from typing import Tuple, List
from sklearn.svm import SVR
import pickle
import os
import csv
from werkzeug import secure_filename
import uuid
from rb.cna.cna_graph import CnaGraph
from rb.utils.rblogger import Logger

logger = Logger.get_logger()


class EssayScoring:


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
    
    """ given a csv, transform it into text files and a csv with filename, grade"""
    def create_files_from_csv(self, path_to_csv_file: str, path_to_folder=".", output_stats_csv='stats.csv'):
        essay_r = csv.reader(open(path_to_csv_file, 'rt', encoding='utf-8'))
        
        csv_stats = [['filename', 'grade']]
        try:
            os.mkdir(path_to_folder)
            logger.info("Directory {} created".format(path_to_folder))
        except FileExistsError:
            logger.info("Directory {} already exists".format(path_to_folder))
        
        for i, row in enumerate(essay_r):
            try:
                grade = float(row[0])
                content = str(row[1])
            except:
                continue
            text_file = secure_filename(str(uuid.uuid4()) + '.txt')
            full_path_text_file = os.path.join(path_to_folder, text_file)
            entry_csv = [text_file, grade]
            csv_stats.append(entry_csv)
            with open(full_path_text_file, 'wt', encoding='utf-8') as f:
                f.write(content)

        with open(os.path.join(path_to_folder, output_stats_csv), 'wt', encoding='utf-8') as csv_essays_eval:
            csv_writer = csv.writer(csv_essays_eval)
            csv_writer.writerows(csv_stats)


    def compute_indices(self, base_folder: str='essays_ro', write_file: str='measurements.csv',
            stats: str='stats.csv', lang: Lang = Lang.RO, nr_docs: int=None) -> List[List]:
        
        indices = []
        if lang is Lang.RO:
            with open('rb/processings/scoring/indices_ro_scoring.txt', 'rt', encoding='utf-8') as f:
                for line in f:
                    indices.append(line.strip())
                    
        vector_model = self.get_vector_model(lang=lang)
            
        all_rows = []
        first_row = ['filename', 'grade']
        essay_r = csv.reader(open(os.path.join(base_folder, stats), 'rt', encoding='utf-8'))

        for i, row in enumerate(essay_r):
            if i == 0:  continue
            if (not (nr_docs is None)) and i > nr_docs:  break
            logger.info('Computing indices for document number {}'.format(i))
            try:
                grade = float(row[1])
                file_name = str(row[0])
            except:
                continue

            with open(os.path.join(base_folder, file_name), 'rt', encoding='utf-8') as f:
                content = f.read()
            
            doc = Document(lang=lang, text=content)
            cna_graph = CnaGraph(docs=doc, models=[vector_model])
            compute_indices(doc=doc, cna_graph=cna_graph)

            row = []
            if len(first_row) == 2:
                for ind in indices:
                    for key, v in doc.indices.items():
                        if str(key) == ind:
                            first_row.append(key)
                all_rows.append(first_row)

            row = [file_name, grade]
            for ind in indices:
                for key, v in doc.indices.items():
                    if str(key) == ind:
                        row.append(v)
            
            if len(row) == len(first_row):
                all_rows.append(row)

        with open(os.path.join(base_folder, write_file), 'wt', encoding='utf-8') as csv_essays_eval:
            csv_writer = csv.writer(csv_essays_eval)
            csv_writer.writerows(all_rows)
        return all_rows
    
    def read_indices(self, base_folder: str = 'essays_ro',
         path_to_csv_file='measurements.csv') -> List[List[float]]:
        essay_r = csv.reader(open(os.path.join(base_folder, path_to_csv_file), 'rt', encoding='utf-8'))
        results = []
        """ first row is the score """
        for i, row in enumerate(essay_r):
            if i == 0:  continue
            results.append(row[1:])
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

        logger.info('Scoring loss: {}'.format(loss))

    def predict(self, content: str, file_to_svr_model: str, lang=Lang.RO) -> float:

        svr = pickle.load(open(file_to_svr_model, "rb"))

        doc = Document(lang=lang, text=content)
        vector_model = self.get_vector_model(lang=lang)
        cna_graph = CnaGraph(docs=doc, models=[vector_model])
        compute_indices(doc=doc, cna_graph=cna_graph)

        indices = []
        if lang is Lang.RO:
            with open('rb/processings/scoring/indices_ro_scoring.txt', 'rt', encoding='utf-8') as f:
                for line in f:
                    indices.append(line.strip())
        values_indices = []
        for ind in indices: 
            for key, v in doc.indices.items():
                if repr(key) == ind:
                    values_indices.append(v)

        grade = svr.predict([values_indices])[0]
        return max(grade - 7, 0.1)

        