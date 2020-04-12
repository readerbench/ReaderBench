from rb.core.lang import Lang
from rb.core.document import Document
from rb.core.text_element_type import TextElementType
from rb.complexity.complexity_index import ComplexityIndex, compute_indices
from rb.similarity.vector_model import VectorModelType, CorporaEnum, VectorModel
from rb.similarity.vector_model_factory import VECTOR_MODELS, create_vector_model
from rb.cna.cna_graph import CnaGraph
from typing import Tuple, List, Dict
import os
import numpy as np
import csv
import uuid
from rb.utils.rblogger import Logger

logger = Logger.get_logger()


class Feedback:


    def __init__(self):
        pass
    
    def get_vector_model(self, lang: Lang = Lang.RO) -> VectorModel:
        global logger
        if lang is Lang.RO:
            vector_model = create_vector_model(Lang.RO, VectorModelType.from_str('word2vec'), "readme")
        elif lang is Lang.EN:
            vector_model = create_vector_model(Lang.EN, VectorModelType.from_str("word2vec"), "coca")
        else:
            logger.error(f'Language {lang.value} is not supported for feedback task')
            vector_model = None
        return vector_model

    def get_used_indices(self) -> Dict[TextElementType, List[str]]:
        indices_files = ['ro_indices_word.txt', 'ro_indices_sent.txt', 'ro_indices_block.txt', 'ro_indices_doc.txt']
        indices_names = {TextElementType.DOC: [],
                         TextElementType.BLOCK: [],
                         TextElementType.SENT: [],
                         TextElementType.WORD: []}
        for i, in_file in enumerate(indices_files):
            lvl = None
            if i == 0:
                lvl = TextElementType.WORD
            elif i == 1:
                lvl = TextElementType.SENT
            elif i == 2:
                lvl = TextElementType.BLOCK
            else:
                lvl = TextElementType.DOC

            with open(os.path.join('rb/processings/readme_feedback', in_file), 'rt', encoding='utf-8') as f:
                for line in f:
                    if len(line.strip()) > 0:
                        indices_names[lvl].append(line.strip())
        return indices_names

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
        return avg - 2.0 * stdev, avg + 2.0 * stdev 

    def compute_extreme_values(self,
        path_to_csv='categories_readme/en_stats.csv', output_file='readme_extreme_values.txt'):
        indices = self.get_used_indices()
        ind_name_to_row = {vv: 0 for v in list(indices.values()) for vv in v}
        ind_name_to_values = {vv: [] for v in list(indices.values()) for vv in v}
        ind_name_to_extreme_values = {vv: (0, 0) for v in list(indices.values()) for vv in v}

        stats = csv.reader(open(path_to_csv, 'rt', encoding='utf-8'))
        for i, row in enumerate(stats):
            if i == 0:
                for i, ind_name in enumerate(row):
                    if ind_name in ind_name_to_row:
                        ind_name_to_row[ind_name] = i
            else:
                for j, ind_value in enumerate(row):
                    for ind_name, ind_row in ind_name_to_row.items():
                        if ind_row == j:
                            ind_name_to_values[ind_name].append(float(ind_value))
                            break
        output_file = open(output_file, 'w')
        for ind_name, values in ind_name_to_values.items():
            ind_name_to_extreme_values[ind_name] = self.compute_thresholds(values)
            print(ind_name, ind_name_to_extreme_values[ind_name], file=output_file)
        return ind_name_to_extreme_values            

    def compute_indices_and_tokenization(self, text, lang: Lang) -> Dict[str, Dict[str, List]]:
        result = {
            TextElementType.WORD.name: {},
            TextElementType.SENT.name: {},
            TextElementType.BLOCK.name: {},
            TextElementType.DOC.name: {}
        }

        KEY_TEXT = "text"
        KEY_DOC = "doc"
        KEY_BLOCK = "block"
        KEY_SENT = "sentence"
        KEY_WORD = "word"
        KEY_VALUES = "values"

        indices = self.get_used_indices()
        doc = Document(lang=lang, text=text)
        vector_model = self.get_vector_model(lang=lang)
        cna_graph = CnaGraph(docs=doc, models=[vector_model])
        compute_indices(doc=doc, cna_graph=cna_graph)
        words, sents, blocks, docs = [], [], [], []
        ind_words, ind_sents, inds_blocks, inds_doc = indices[TextElementType.WORD],\
                  indices[TextElementType.SENT],  indices[TextElementType.BLOCK], indices[TextElementType.DOC]

        # doc
        d_ind_list = []
        for ind in indices[TextElementType.DOC]:
            for ind_name, ind_v in doc.indices.items():
                if repr(ind_name) == ind:
                    d_ind_list.append(ind_v)
        docs.append({KEY_TEXT: doc.text, KEY_BLOCK: 0, KEY_SENT: 0,
                             KEY_WORD: 0, KEY_VALUES: d_ind_list, KEY_DOC: 0})
        t_doc = []
        for i_block, block in enumerate(doc.get_blocks()):
            # block 
            b_ind_list = []
            for ind in indices[TextElementType.BLOCK]:
                for ind_name, ind_v in block.indices.items():
                    if repr(ind_name) == ind:
                        b_ind_list.append(ind_v)
            blocks.append({KEY_TEXT: block.text, KEY_BLOCK: i_block, KEY_SENT: 0,
                             KEY_WORD: 0, KEY_VALUES: b_ind_list, KEY_DOC: 0})

            t_block = []
            for i_sent, sent in enumerate(block.get_sentences()):
                # sent
                s_ind_list = []
                for ind in indices[TextElementType.SENT]:
                    for ind_name, ind_v in sent.indices.items():
                        if repr(ind_name) == ind:
                            s_ind_list.append(ind_v)
                sents.append({
                    KEY_TEXT: sent.text, KEY_BLOCK: i_block, KEY_SENT: i_sent, KEY_VALUES: s_ind_list,
                    KEY_WORD: 0, KEY_DOC: 0})

                t_sent = []
                for i_word, word in enumerate(sent.get_words()):
                    # word
                    w_ind_list = []
                    for ind in indices[TextElementType.WORD]:
                        for ind_name, ind_v in word.indices.items():
                            if repr(ind_name) == ind:
                                w_ind_list.append(ind_v)
                    words.append({KEY_TEXT: word.text, KEY_BLOCK: i_block, KEY_SENT: i_sent,
                             KEY_WORD: i_word, KEY_VALUES: w_ind_list, KEY_DOC: 0})
                    t_sent.append(word.text)
                t_block.append(t_sent)
            t_doc.append(t_block)

        result[TextElementType.DOC.name] = {
            'indices': indices[TextElementType.DOC],
            'results': docs
        }
        result[TextElementType.BLOCK.name] = {
            'indices': indices[TextElementType.BLOCK],
            'results': blocks
        }
        result[TextElementType.SENT.name] = {
            'indices': indices[TextElementType.SENT],
            'results': sents
        }
        result[TextElementType.WORD.name] = {
            'indices': indices[TextElementType.WORD],
            'results': words
        }

        result['TOKENIZATION'] = t_doc

        ff = open('debug.txt', 'w')
        print(result, file=ff)

        return result