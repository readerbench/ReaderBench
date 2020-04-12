from rb.processings.pipeline.estimator import Classifier
from rb.processings.pipeline.dataset import Dataset, Task
from typing import List, Dict
from sklearn.neural_network import MLPClassifier

class MLP(Classifier):
    def __init__(self, dataset: Dataset, tasks: List[Task], params: Dict[str, str]):
        super().__init__(dataset, tasks, params)
        self.model = MLPClassifier(hidden_layer_sizes=params["hidden_layer_sizes"])

    @classmethod
    def parameters(cls):
        return {
            "hidden_layer_sizes": [32, 64, 128]
        }