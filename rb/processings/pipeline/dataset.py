
import copy
import csv
import pickle
import random
from enum import Enum, auto
from typing import Dict, List, Tuple

import numpy as np
from rb.complexity.complexity_index import ComplexityIndex
from rb.core.document import Document
from rb.core.lang import Lang


class TargetType(Enum):
    FLOAT = auto()
    INT = auto()
    STR = auto()

class Task:
    def __init__(self, type: TargetType, values: List[str]):
        if type is TargetType.FLOAT:
            self.values = [float(val) for val in values]
            self.min = min(self.values)
            self.max = max(self.values)
        elif type is TargetType.INT:
            self.values = [int(val) for val in values]
            unique = len(set(self.values))
            if max(self.values) - min(self.values) + 1 != unique:
                type = TargetType.STR
        if type is TargetType.STR:
            self.values = values
            self.classes = list({val for val in self.values})
            self.index = {c: i for i, c in enumerate(self.classes)}
        self.type = type
        self.mask: List[bool] = []
        self.train_values = []
        self.dev_values = []

    def _get_targets(self, values) -> List:
        if self.type is TargetType.STR:
            return [self.index[val] for val in values]
        else:
            return values
    
    def get_train_targets(self) -> List:
        return self._get_targets(self.train_values)

    def get_dev_targets(self) -> List:
        return self._get_targets(self.dev_values)

class Dataset:

    def __init__(self, lang: Lang, names: List[str], texts: List[str], labels: List[List[str]]):
        self.lang = lang
        self.names = names
        self.texts = texts
        self.tasks = self.convert_labels(labels)
        self.train_texts: List[str] = []
        self.dev_texts: List[str] = []
        self.features: List[ComplexityIndex] = []
        self.train_features: List[Dict[ComplexityIndex, float]] = []
        self.dev_features: List[Dict[ComplexityIndex, float]] = []
        self.all_features: List[Dict[ComplexityIndex, float]] = []
        self.train_indices: List[int] = []
        self.dev_indices: List[int] = []
    
    def split(self, dev_ratio: float):
        indices = list(range(len(self.texts)))
        random.shuffle(indices)
        n_dev = int(dev_ratio * len(self.texts))
        self.dev_indices = indices[:n_dev]
        self.train_indices = indices[n_dev:]
        self.expand()
    
    def expand(self):
        self.train_texts = [self.texts[index] for index in self.train_indices]
        self.dev_texts = [self.texts[index] for index in self.dev_indices]
        self.train_features = [self.all_features[index] for index in self.train_indices]
        self.dev_features = [self.all_features[index] for index in self.dev_indices]
        for task in self.tasks:
            task.train_values = [task.values[index] for index in self.train_indices]
            task.dev_values = [task.values[index] for index in self.dev_indices]
            task.values = None
    
    def reduce_size(self):
        self.all_features = [
            {
                feature: indices[feature]
                for feature in self.features
            }
            for indices in self.all_features
        ]
        self.train_features = [
            {
                feature: indices[feature]
                for feature in self.features
            }
            for indices in self.train_features
        ]
        self.dev_features = [
            {
                feature: indices[feature]
                for feature in self.features
            }
            for indices in self.dev_features
        ]
    
    def normalize_features(self):
        self.normalized_train_features = [{} for _ in range(len(self.train_features))]
        self.normalized_dev_features = [{} for _ in range(len(self.dev_features))]
        for feature in self.features:
            values = [indices[feature] for indices in self.train_features]
            min_value = min(values)
            max_value = max(values)
            for indices, val in zip(self.normalized_train_features, values):
                indices[feature] = (val - min_value) / (max_value - min_value)
            for indices, val in zip(self.normalized_dev_features, [indices[feature] for indices in self.dev_features]):
                indices[feature] = (val - min_value) / (max_value - min_value)
            
    def convert_labels(self, labels: List[List[str]]) -> List[Task]:
        values = zip(*labels)
        tasks = []
        for targets in values:
            if all(is_double(target) for target in targets):
                tasks.append(Task(TargetType.FLOAT, targets))
            elif all(is_int(target) for target in targets):
                tasks.append(Task(TargetType.INT, targets))
            else:
                tasks.append(Task(TargetType.STR, targets))
        return tasks

    def save_features(self, filename: str):
        with open(filename, "wt", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter=",")
            writer.writerow(["name"] + [repr(index) for index in self.features])
            for name, features in zip(self.names, self.all_features):
                writer.writerow([name] + [features[index] for index in self.features])

    def save(self, filename: str):
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load_features(filename: str) -> List[List[float]]:
        with open(filename, "rt", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter=",")
            header = next(reader)
            return [[float(x) for x in row[1:]] for row in reader]

    @staticmethod
    def load(filename: str) -> "Dataset":
        with open(filename, "rb") as f:
            return pickle.load(f)

def is_double(value: str) -> bool:
    try:
        float(value)
    except ValueError:
        return False
    return True

def is_int(value: str) -> bool:
    try:
        int(value)
    except ValueError:
        return False
    return True
