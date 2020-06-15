from typing import Dict, List

import numpy as np
from rb.complexity.complexity_index import ComplexityIndex
from rb.core.document import Document
from rb.processings.pipeline.dataset import Dataset, Task
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, make_scorer, mean_squared_error


class Estimator:
    def __init__(self, dataset: Dataset, tasks: List[Task], params: Dict[str, str]):
        self.dataset = dataset
        self.tasks = tasks
        self.model: BaseEstimator = None
        self.scoring = "accuracy"

    def predict(self, indices: Dict[ComplexityIndex, float]):
        return self.model.predict(self.construct_input(indices))

    def construct_input(self, indices: Dict[ComplexityIndex, float]) -> np.ndarray:
        result = [indices[feature]
                  for i, feature in enumerate(self.dataset.features)
                  if self.tasks[0].mask[i]]
        return np.array(result)

    def cross_validation(self, n=5) -> float:
        x = [self.construct_input(indices) for indices in self.dataset.normalized_train_features]
        y = self.tasks[0].get_train_targets()
        scores = cross_val_score(self.model, x, y, scoring=make_scorer(self.scoring), cv=n)
        return scores.mean()
    
    def evaluate(self) -> float:
        x = [self.construct_input(indices) for indices in self.dataset.normalized_train_features]
        y = self.tasks[0].get_train_targets()
        self.model.fit(x, y)
        x = [self.construct_input(indices) for indices in self.dataset.normalized_dev_features]
        y = self.tasks[0].get_dev_targets()
        predicted = self.model.predict(x)
        return self.scoring(y, predicted)

    @classmethod
    def parameters(cls) -> Dict[str, List]:
        return {}
    
    @classmethod
    def valid_config(cls, config) -> bool:
        return True

class Classifier(Estimator):
    def __init__(self, dataset: Dataset, tasks: List[Task], params: Dict[str, str]):
        super().__init__(dataset, tasks, params)
        self.scoring = accuracy_score

class Regressor(Estimator):
    def __init__(self, dataset: Dataset, tasks: List[Task], params: Dict[str, str]):
        super().__init__(dataset, tasks, params)
        self.scoring = mean_squared_error

