from rb.processings.pipeline.estimator import Classifier
from rb.processings.pipeline.dataset import Dataset, Task
from typing import List, Dict
from sklearn.ensemble import RandomForestClassifier

class RandomForest(Classifier):
    def __init__(self, dataset: Dataset, tasks: List[Task], params: Dict[str, str]):
        super().__init__(dataset, tasks, params)
        self.model = RandomForestClassifier(n_estimators=params["n_estimators"])
        self.n_estimators: int = params["n_estimators"]
    
    @classmethod
    def parameters(cls):
        return {
            "n_estimators": [32, 64, 96, 128]
        }
        
    def __str__(self):
        return f"Random forest - {self.n_estimators} estimators"