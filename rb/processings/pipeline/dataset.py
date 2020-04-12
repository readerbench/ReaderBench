
from typing import List, Tuple
from enum import Enum, auto
import random
import copy
from rb.core.document import Document
from rb.complexity.complexity_index import ComplexityIndex
import csv
import pickle

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

    def __init__(self, docs: List[str], labels: List[List[str]]):
        self.docs = docs
        self.tasks = self.convert_labels(labels)
        self.train_texts: List[str] = []
        self.dev_texts: List[str] = []
        self.train_docs: List[Document] = []
        self.dev_docs: List[Document] = []
        self.features: List[ComplexityIndex] = []
        self.split(0.2)
        
    
    def split(self, dev_ratio: float):
        indices = list(range(len(self.docs)))
        random.shuffle(indices)
        n_dev = int(dev_ratio * len(self.docs))
        dev_indices = indices[:n_dev]
        train_indices = indices[n_dev:]
        self.train_texts = [self.docs[index] for index in train_indices]
        self.dev_texts = [self.docs[index] for index in dev_indices]
        for task in self.tasks:
            task.train_values = [task.values[index] for index in train_indices]
            task.dev_values = [task.values[index] for index in dev_indices]
            task.values = None
        
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

    # def separate_tasks(self) -> List["Dataset"]:
    #     result = []
    #     for train_task, dev_task in zip(self.train_tasks, self.dev_tasks):
    #         new_dataset = copy.copy(self)
    #         new_dataset.train_tasks = [train_task]
    #         new_dataset.dev_tasks = [dev_task]
    #         result.append(new_dataset)
    #     return result

    def save_features(self, filename: str):
        with open(filename, "wt", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter=",")
            writer.writerow([repr(index) for index in self.features])
            for doc in self.train_docs:
                writer.writerow([doc.indices[index] if index in doc.indices else "" for index in self.features])

    def save(self, filename: str):
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load_features(filename: str) -> List[List[float]]:
        with open(filename, "rt", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter=",")
            header = next(reader)
            return [[float(x) for x in row[1:]] for row in reader]

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

        


