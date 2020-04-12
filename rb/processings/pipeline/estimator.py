from rb.processings.pipeline.dataset import Dataset, Task
from rb.core.document import Document
import numpy as np
from typing import List, Dict
from sklearn.model_selection import cross_val_score
from sklearn.base import BaseEstimator

class Estimator:
    def __init__(self, dataset: Dataset, tasks: List[Task], params: Dict[str, str]):
        self.dataset = dataset
        self.tasks = tasks
        self.model: BaseEstimator = None
        self.scoring = "accuracy"

    def predict(self, doc: Document):
        pass

    def construct_input(self, doc: Document) -> np.ndarray:
        result = [doc.indices[feature]
                  for i, feature in enumerate(self.dataset.features)
                  if self.tasks[0].mask[i]]
        return np.array(result)

    def cross_validation(self, n=5) -> float:
        x = [self.construct_input(doc) for doc in self.dataset.train_docs]
        y = self.tasks[0].get_targets()
        scores = cross_val_score(self.model, x, y, scoring=self.scoring)
        return scores.mean()

    @classmethod
    def parameters(cls) -> Dict[str, List]:
        return {}

class Classifier(Estimator):
    def __init__(self, dataset: Dataset, tasks: List[Task], params: Dict[str, str]):
        super().__init__(dataset, tasks, params)
        self.scoring = "accuracy"

class Regressor(Estimator):
    def __init__(self, dataset: Dataset, tasks: List[Task], params: Dict[str, str]):
        super().__init__(dataset, tasks, params)
        self.scoring = "neg_root_mean_squared_error"


