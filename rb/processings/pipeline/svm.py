from rb.processings.pipeline.estimator import Classifier
from rb.processings.pipeline.dataset import Dataset, Task
from typing import List, Dict
from sklearn import svm

class SVM(Classifier):
    def __init__(self, dataset: Dataset, tasks: List[Task], params: Dict[str, str]):
        super().__init__(dataset, tasks, params)
        self.model = svm.SVC(kernel=params["kernel"])

    @classmethod
    def parameters(cls):
        return {
            "kernel": ["rbf", "poly", "sigmoid"]
        }