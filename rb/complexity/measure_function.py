from enum import Enum
from functools import partial
from statistics import mean, stdev
from typing import List

def average(elements: List[float]) -> float:
    if len(elements) == 0:
        return 0
    return mean(elements)

def standard_deviation(elements: List[float]) -> float:
    if len(elements) == 0:
        return 0
    elif len(elements) == 1:
        return 0
    return stdev(elements)

class MeasureFunction(Enum):
    AVG = average
    STDEV = standard_deviation