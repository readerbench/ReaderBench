import csv
import math
import os
from heapq import heapify, heappop
from typing import List, Dict, Iterable

from rb.cna.cna_graph import CnaGraph
from rb.complexity.complexity_index import compute_indices, ComplexityIndex
from rb.core.document import Document
from rb.core.lang import Lang
from rb.processings.pipeline.dataset import Dataset, TargetType, Task
from rb.similarity.vector_model_factory import get_default_model
from scipy.stats import f_oneway, pearsonr
from rb.processings.pipeline.estimator import Estimator
from rb.processings.pipeline.svm import SVM
from rb.processings.pipeline.svr import SVR
from rb.processings.pipeline.ridge_regression import RidgeRegression
from rb.processings.pipeline.mlp import MLP
from rb.processings.pipeline.random_forrest import RandomForest

CLASSIFIERS = [SVM, RandomForest, MLP]
REGRESSORS = [SVR, RidgeRegression]

def construct_documents(dataset: List[str], lang: Lang) -> List[Document]:
    model = get_default_model(lang)
    result = []
    for text in dataset:
        doc = Document(lang, text)
        cna_graph = CnaGraph(docs=doc, models=[model])
        compute_indices(doc=doc, cna_graph=cna_graph)
        result.append(doc)
    return result

def filter_rare(dataset: Dataset):
    features = []
    for index in dataset.features:
        values = [doc.indices[index] for doc in dataset.train_docs]
        zeros = sum(1 for val in values if val == 0)
        if zeros / len(values) < 0.2:
            features.append(index)
    dataset.features = features

def preprocess(folder: str, targets_file: str, limit: int = None) -> Dataset:
    files = {filename.replace(".txt", "").strip(): open(os.path.join(folder, filename), "rt", encoding='utf-8', errors='ignore').read().strip() 
             for filename in os.listdir(folder)
             if not filename.startswith(".")}
    texts = []
    targets = []
    with open(targets_file, "rt", encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=',')
        header = next(reader)
        for line in reader:
            filename = line[0].replace(".txt", "").strip()
            if filename not in files:
                print(filename)
                continue
            texts.append(files[filename])
            targets.append(line[1:])
    dataset = Dataset(texts, targets)
    if limit is not None:
        dataset.train_texts = dataset.train_texts[:limit]
        dataset.dev_texts = dataset.dev_texts[:limit]
        for task in dataset.tasks:
            task.train_values = task.train_values[:limit]
            task.dev_values = task.dev_values[:limit]
    
    dataset.train_docs = construct_documents(dataset.train_texts, Lang.EN)
    dataset.dev_docs = construct_documents(dataset.dev_texts, Lang.EN)
    dataset.features = list(dataset.train_docs[0].indices.keys())
    filter_rare(dataset)
    for task in dataset.tasks:
        remove_colinear(dataset, task)
    return dataset
    # dataset.save_features("features.csv")

def correlation_with_targets(feature: ComplexityIndex, dataset: Dataset, task: Task) -> float:
    values = [doc.indices[feature] for doc in dataset.train_docs]
    if task.type is TargetType.FLOAT:
        corr, p = pearsonr(values, task.train_values)
        return abs(corr)
    values_per_class = {}
    for val, target in zip(values, task.train_values):
        if target not in values_per_class:
            values_per_class[target] = []
        values_per_class[target].append(val)
    return f_oneway(*values_per_class.values())

def remove_colinear(dataset: Dataset, task: Task) -> None:
    # features = list(zip(*Dataset.load_features("features.csv")))
    heap = []
    for i, a in enumerate(dataset.features[:-1]):
        for j, b in enumerate(dataset.features[i+1:]):
            values_a = [doc.indices[a] for doc in dataset.train_docs]
            values_b = [doc.indices[b] for doc in dataset.train_docs]
            corr, p = pearsonr(values_a, values_b)
            if math.isnan(corr):
                continue
            heap.append((-corr, i, j))
    heapify(heap)
    
    correlations = [correlation_with_targets(feature, dataset, task) 
                    for feature in dataset.features]
    task.mask = [True] * len(dataset.features)
    while len(heap) > 0:
        inv_corr, i, j = heappop(heap)
        if not task.mask[i] or not task.mask[j]:
            continue
        if inv_corr < -0.9:
            if correlations[i] > correlations[j]:
                task.mask[j] = False
            else:
                task.mask[i] = False


def next_config(parameters: Dict[str, List]) -> Iterable[Dict[str, str]]:
    solution  = [-1] * len(parameters)
    keys = list(parameters.keys())
    current = 0
    while current >= 0:
        if current == len(parameters):
            yield {keys[i]: parameters[keys[i]][j] for i, j in enumerate(solution)}
            current -= 1
        if solution[current] == len(parameters[keys[current]]) - 1:
            solution[current] = -1
            current -= 1
            continue
        solution[current] += 1
        current += 1
            

def grid_search(dataset: Dataset, task: Task) -> Estimator:
    if task.type is TargetType.FLOAT:
        estimators = REGRESSORS
    else:
        estimators = CLASSIFIERS
    for estimator in estimators:
        parameters = estimator.parameters()
        for config in next_config(parameters):
            model = estimator(dataset, [task], config)
            acc = model.cross_validation()
            print("{} - {}".format(estimator, acc))
            
           
    