from rb.core.lang import Lang
from rb.core.text_element_type import TextElementType
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

