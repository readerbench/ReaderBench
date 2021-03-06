from rb.processings.pipeline.estimator import Regressor
from rb.processings.pipeline.dataset import Dataset, Task
from typing import List, Dict
from sklearn import svm

class SVR(Regressor):
    def __init__(self, dataset: Dataset, tasks: List[Task], params: Dict[str, str]):
        super().__init__(dataset, tasks, params)
        self.model = svm.SVR(gamma='scale', kernel=params["kernel"], degree=params["degree"])
        self.kernel = params["kernel"]
        self.degree = params["degree"]
        
    @classmethod
    def parameters(cls):
        return {
            "kernel": ["rbf", "poly", "sigmoid"],
            "degree": [2,3,4,5],
        }
    
    @classmethod
    def valid_config(cls, config):
        return config["kernel"] == "poly" or config["degree"] == 3
    
    def __str__(self):
        return f"SVR - {self.kernel}" + (f"({self.degree})" if self.kernel == "poly" else "")