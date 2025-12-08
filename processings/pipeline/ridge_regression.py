from rb.processings.pipeline.estimator import Regressor
from rb.processings.pipeline.dataset import Dataset, Task
from typing import List, Dict
from sklearn.linear_model import Ridge

class RidgeRegression(Regressor):
    def __init__(self, dataset: Dataset, tasks: List[Task], params: Dict[str, str]):
        super().__init__(dataset, tasks, params)
        self.model = Ridge(alpha=params["alpha"])
        self.alpha = params["alpha"]

    @classmethod
    def parameters(cls):
        return {
            "alpha": [0.5, 1, 1.5]
        }
    
    def __str__(self):
        return f"Ridge regression - alpha={self.alpha}"