import csv
import math
import os
from heapq import heapify, heappop
from typing import List

from rb.cna.cna_graph import CnaGraph
from rb.complexity.complexity_index import compute_indices
from rb.core.document import Document
from rb.core.lang import Lang
from rb.processings.pipeline.dataset import Dataset, TargetType
from rb.similarity.vector_model_factory import get_default_model
from scipy.stats import f_oneway, pearsonr


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

def preprocess(folder: str, targets_file: str) -> Dataset:
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
    dataset.train_docs = construct_documents(dataset.train_texts[:10], Lang.EN)
    dataset.dev_docs = construct_documents(dataset.dev_texts[:10], Lang.EN)
    for task in dataset.train_tasks + dataset.dev_tasks:
        task.values = task.values[:10]
    dataset.features = list(dataset.train_docs[0].indices.keys())
    filter_rare(dataset)
    return dataset
    # dataset.save_features("features.csv")

def correlation_with_targets(feature: int, dataset: Dataset) -> float:
    values = [doc.indices[dataset.features[feature]] for doc in dataset.train_docs]
    task = dataset.train_tasks[0]
    if task.type is TargetType.FLOAT:
        corr, p = pearsonr(values, task.values)
        return abs(corr)
    values_per_class = {}
    for val, target in zip(values, task.values):
        if target not in values_per_class:
            values_per_class[target] = []
        values_per_class[target].append(val)
    return f_oneway(values_per_class.values())

def remove_colinear(dataset: Dataset) -> None:
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
    removed = set()
    while len(heap) > 0:
        inv_corr, i, j = heappop(heap)
        if i in removed or j in removed:
            continue
        if inv_corr < -0.9:
            if correlation_with_targets(i, dataset) > correlation_with_targets(j, dataset):
                removed.add(j)
            else:
                removed.add(i)
    dataset.features = [feature for i, feature in enumerate(dataset.features) if i not in removed]
