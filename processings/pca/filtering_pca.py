from rb.core.lang import Lang
from rb.core.document import Document
from rb.complexity.complexity_index import ComplexityIndex, compute_indices
from rb.complexity.cohesion.adj_cohesion import AdjCohesion
from rb.similarity.word2vec import Word2Vec
from rb.cna.cna_graph import CnaGraph
from typing import Tuple, List
from sklearn.svm import SVR
import pickle
import os
import csv
from rb.utils.rblogger import Logger
from typing import List, Tuple
import numpy as np

logger = Logger.get_logger()

class Fluctuations:

    def __init__(self):
        pass
    