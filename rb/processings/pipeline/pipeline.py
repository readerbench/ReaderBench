import csv
import math
import os
from heapq import heapify, heappop
from typing import Dict, Iterable, List, Tuple

import numpy as np
from joblib import Parallel, delayed
from rb.cna.cna_graph import CnaGraph
from rb.complexity.complexity_index import ComplexityIndex, compute_indices
from rb.core.document import Document
from rb.core.lang import Lang
from rb.core.meta_document import MetaDocument
from rb.processings.pipeline.bert_classifier import BertClassifier
from rb.processings.pipeline.dataset import Dataset, TargetType, Task
from rb.processings.pipeline.estimator import Estimator
from rb.processings.pipeline.mlp import MLPClassifier, MLPRegressor
from rb.processings.pipeline.random_forrest import RandomForest
from rb.processings.pipeline.ridge_regression import RidgeRegression
from rb.processings.pipeline.svm import SVM
from rb.processings.pipeline.svr import SVR
from rb.similarity.vector_model import VectorModel
from rb.similarity.vector_model_factory import get_default_model
from rb.utils.rblogger import Logger
from scipy.stats import f_oneway, pearsonr

CLASSIFIERS = [SVM, RandomForest, MLPClassifier]
REGRESSORS = [SVR, RidgeRegression, MLPRegressor]
    
logger = Logger.get_logger()
        
 
def construct_document(lang: Lang, text: str) -> MetaDocument:
    sections = ["\n".join(line for line in section.split("\n") if len(line) > 50) for section in text.split("\n\n")]
    return MetaDocument(lang, sections)
    
    
def compute_features(doc: MetaDocument):
    model = get_default_model(doc.lang)
    cna_graph = CnaGraph(docs=doc, models=[model])
    for section in doc.components:
        compute_indices(doc=section, cna_graph=cna_graph)     
    return {
        feature: np.mean([section.indices[feature] for section in doc.components]) 
        for feature in doc.components[0].indices.keys()
    }
        
def construct_documents(dataset: List[str], lang: Lang) -> List[Dict[ComplexityIndex, float]]:
    logger.info("Constructing documents..")
    docs = Parallel(n_jobs=1)( \
        delayed(construct_document)(lang, text) \
        for text in dataset)
    logger.info("Constructing graphs..")
    return Parallel(n_jobs=-1, prefer="processes")( \
        delayed(compute_features)(doc) \
        for doc in docs)

    
def filter_rare(dataset: Dataset):
    features = []
    for index in dataset.features:
        values = [indices[index] for indices in dataset.train_features]
        zeros = sum(1 for val in values if val == 0)
        if zeros / len(values) < 0.2:
            features.append(index)
    dataset.features = features

def preprocess(folder: str, targets_file: str, lang: Lang, limit: int = None) -> Dataset:
    files = {filename.replace(".txt", "").strip(): open(os.path.join(folder, filename), "rt", encoding='utf-8', errors='ignore').read().strip() 
             for filename in os.listdir(folder)
             if not filename.startswith(".")}
    names = []
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
            names.append(filename)
            texts.append(files[filename])
            targets.append(line[1:])
            if limit is not None and len(names) == limit:
                break
    dataset = Dataset(lang, names, texts, targets)
    
    dataset.all_features = construct_documents(dataset.texts, lang)
    dataset.features = list(dataset.all_features[0].keys())
    dataset.split(0.2)
    filter_rare(dataset)
    logger.info("Removing colinear..")
    for task in dataset.tasks:
        remove_colinear(dataset, task)
    return dataset
    # dataset.save_features("features.csv")

def correlation_with_targets(feature: ComplexityIndex, dataset: Dataset, task: Task) -> float:
    values = [indices[feature] for indices in dataset.train_features]
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
    heap = []
    for i, a in enumerate(dataset.features[:-1]):
        for j, b in enumerate(dataset.features[i+1:]):
            values_a = [indices[a] for indices in dataset.train_features]
            values_b = [indices[b] for indices in dataset.train_features]
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


def next_config(estimator, parameters: Dict[str, List]) -> Iterable[Dict[str, str]]:
    solution  = [-1] * len(parameters)
    keys = list(parameters.keys())
    current = 0
    while current >= 0:
        if current == len(parameters):
            config = {keys[i]: parameters[keys[i]][j] for i, j in enumerate(solution)}
            if estimator.valid_config(config):
                yield config
            current -= 1
        if current < 0:
            break
        if solution[current] == len(parameters[keys[current]]) - 1:
            solution[current] = -1
            current -= 1
            continue
        solution[current] += 1
        current += 1
            

def grid_search(dataset: Dataset, task: Task) -> Estimator:
    if task.type is TargetType.FLOAT:
        estimators = REGRESSORS
        default_best = (None, float('inf'))
        better = lambda a, b: a < b
    else:
        estimators = CLASSIFIERS
        default_best = (None, 0)
        better = lambda a, b: a > b
    results = []    
    for estimator in estimators:
        parameters = estimator.parameters()
        best = default_best
        for config in next_config(estimator, parameters):
            model = estimator(dataset, [task], config)
            acc = model.cross_validation()
            if better(acc, best[1]):
                best = (model, acc)
            print(f"{model} - {acc}")
        model = best[0]
        score = model.evaluate()
        results.append((model, score))
    return results

def bert_grid_search(dataset: Dataset, use_indices=True, use_mask=True, shared=True) -> Tuple[Estimator, List[float]]:
    parameters = BertClassifier.parameters()
    best = (None, 0, float('inf'))
    for config in next_config(BertClassifier, parameters):
        model = BertClassifier(dataset, dataset.tasks, config, use_indices, use_mask, shared)
        epoch, loss = model.cross_validation()
        print(f"{config}: loss={loss}")
        if loss < best[2]:
            best = (model, epoch, loss)
    model, epoch, loss = best
    print(f"Best model after {epoch} epochs with loss={loss}")
    model.initialize()
    scores = model.train(epoch)
    return model, scores

def evaluate_bert_models(dataset: Dataset) -> List[Tuple[Estimator, List[float]]]:
    results = []
    results.append(bert_grid_search(dataset, use_indices=False, shared=False))
    results.append(bert_grid_search(dataset, use_indices=True, shared=False, use_mask=False))
    results.append(bert_grid_search(dataset, use_indices=True, use_mask=True, shared=False))
    results.append(bert_grid_search(dataset, use_indices=True, use_mask=False, shared=True))
    results.append(bert_grid_search(dataset, use_indices=True, use_mask=True, shared=True))
    return results
