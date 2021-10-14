from enum import Enum
from functools import partial
from statistics import mean, stdev
from typing import List

def average(elements: List[float]) -> float:
    elements = [elem for elem in elements if elem is not None]
    if len(elements) == 0:
        return None
    return mean(elements)

def standard_deviation(elements: List[float]) -> float:
    elements = [elem for elem in elements if elem is not None]
    if len(elements) == 0:
        return None
    elif len(elements) == 1:
        return 0
    return stdev(elements)

def maximum(elements: List[float]) -> float:
    elements = [elem for elem in elements if elem is not None]
    if len(elements) == 0:
        return None
    return max(elements)


class MeasureFunction(Enum):
    AVG = average
    STDEV = standard_deviation
    MAX = maximum
    IDENTITY = lambda x: x[0]