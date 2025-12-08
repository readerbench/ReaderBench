import numpy as np
from numpy.linalg import norm

class Vector:

    def __init__(self, values: np.ndarray):
        self.values = values
        self.norm: float = norm(values)
        self.size: int = np.size(values)
    
    @staticmethod
    def cosine_similarity(a: "Vector", b: "Vector") -> float:
        return np.dot(a.values, b.values) / (a.norm * b.norm)

    def __len__(self):
        return self.size

    def __eq__(self, value):
        if not isinstance(value, Vector):
            return False
        return self.values == value.values

    def __hash__(self):
        return self.values.__hash__()

    def __repr__(self):
        return self.values.__repr__()