from rb.processings.pipeline.estimator import Classifier, Regressor
from rb.processings.pipeline.dataset import Dataset, Task
from typing import List, Dict
from sklearn import neural_network

class MLPClassifier(Classifier):
    def __init__(self, dataset: Dataset, tasks: List[Task], params: Dict[str, str]):
        super().__init__(dataset, tasks, params)
        self.model = neural_network.MLPClassifier(hidden_layer_sizes=params["hidden_layer_sizes"])
        self.hidden = params["hidden_layer_sizes"]
        
    @classmethod
    def parameters(cls):
        return {
            "hidden_layer_sizes": [32, 64, 96, 128]
        }
    
    def __str__(self):
        return f"MLP - {self.hidden}"

class MLPRegressor(Regressor):
    def __init__(self, dataset: Dataset, tasks: List[Task], params: Dict[str, str]):
        super().__init__(dataset, tasks, params)
        self.model = neural_network.MLPRegressor(hidden_layer_sizes=params["hidden_layer_sizes"])
        self.hidden = params["hidden_layer_sizes"]
        
    @classmethod
    def parameters(cls):
        return {
            "hidden_layer_sizes": [32, 64, 96, 128]
        }
    
    def __str__(self):
        return f"MLP - {self.hidden}"