import heapq
import math
import os
import numpy as np
from collections import defaultdict
from functools import reduce
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, Iterable, List, Union

from rb.utils.rblogger import Logger
from rb.core.lang import Lang
from rb.parser.spacy_parser import SpacyParser
from rb.core.text_element import TextElement
from rb.core.sentence import Sentence
from rb.core.document import Document
from rb.similarity.vector_model_factory import get_default_model
from rb.similarity.vector_model import VectorModel
import itertools
from sklearn.cluster import KMeans

logger = Logger.get_logger()
logger.propagate = False


def process(sentences: List[Sentence], model: VectorModel, k_ratio: float = 0.2):
    """
    :param tokenized_sentences: every sentence is represented as a list of tokens (strings or word embeddings)
    :param k_ratio: represents the ratio between the number of clusters and the number of docs
    :return: tuple(similarity matrix, dictionary of clusters)
    """
    n = len(sentences)
    similarity_matrix = np.zeros([n, n])
    for i, j in itertools.combinations(range(n), 2):
        sim = model.similarity(sentences[i], sentences[j])
        similarity_matrix[i, j] = similarity_matrix[j, i] = sim
    
    assert (0 < k_ratio <= 1), "A value between 0 and 1 must be provided for k_ratio"
    no_clusters = math.ceil(k_ratio * n)
    km = KMeans(n_clusters=no_clusters, max_iter=20).fit(similarity_matrix)
    clusters = km.labels_

    """
    Note: 'clusters' is a list of size n which has values from 0 to k-1 and maps every
    sentence index to the belonging cluster index
    e.g. clusters=[1 1 0 1 1 0 1 0 1] for k=2
    """
    clusters_dict = {}
    for i, cluster in enumerate(clusters):
        if cluster not in clusters_dict:
            clusters_dict[cluster] = []
        clusters_dict[cluster].append(i)
    return similarity_matrix, clusters_dict

def coverage_function(similarity_matrix, summary_indices: List, alpha_constant: int = 5) -> float:
    coverage_result = 0.0
    if len(summary_indices) == 0:
        return coverage_result
    
    n = len(similarity_matrix)
    alpha = 1.0 * alpha_constant / n
    for i in range(n):
        summary_coverage = sum([similarity_matrix[i][j] for j in summary_indices])
        corpus_coverage = sum(similarity_matrix[i])
        coverage_result += min(summary_coverage, alpha * corpus_coverage)
    """
    Note: The explanation behind min formula is that the same value is returned
    when i becomes saturated or is sufficiently covered by summary.
    """
    return coverage_result

def diversity_reward_function(similarity_matrix, clusters: Dict, summary_indices: List) -> float:
    diversity_reward = 0.0
    if len(summary_indices) == 0:
        return diversity_reward

    no_clusters = len(clusters)
    for k in range(no_clusters):
        common_indices = list(set(summary_indices) & set(clusters[k]))
        if len(common_indices) == 0:
            continue
        cluster_reward = sum(max(sum(similarity_matrix[j]) / len(similarity_matrix[j]), 0)  for j in common_indices)
        diversity_reward += math.sqrt(cluster_reward)
    
    return diversity_reward
    
def greedy_submodular_maximization(similarity_matrix, clusters, costs_dict, summary_size, alpha, tradeoff_coefficient, word_count=None):
    n = len(similarity_matrix)
    prev_score = 0

    G = set()
    U = list(range(n))
    
    while len(U) > 0:
        temp_scores = []
        for l in U:
            # c_l = costs_dict[l]
            temp_summary_indices = list(G) + [l]

            coverage_result = coverage_function(similarity_matrix, temp_summary_indices, alpha)
            diversity_reward = diversity_reward_function(similarity_matrix, clusters, temp_summary_indices)
            current_score = coverage_result + tradeoff_coefficient * diversity_reward

            win = (current_score - prev_score)# / pow(c_l, 0.3)
            temp_scores += [(win, l, current_score)]

        (best_win, k, score) = max(temp_scores, key=lambda t: t[0])
        # if sum([costs_dict[i] for i in G]) + costs_dict[k] <= word_count:
        if len(G) + 1 <= summary_size:
            if best_win >= 0:
                G.add(k)
                prev_score = score
        else:
            break
        U = [i for i in U if i != k]
    return sorted(G)

def summarize(doc: Union[str, TextElement], lang: Lang=None, ratio=0.2, word_count=None, tradeoff_coefficient=5, k_ratio=0.1, alpha=3) -> List[Sentence]:
    if isinstance(doc, str):
        assert lang is not None, "lang parameter is required for str docs"
        doc = Document(lang, doc)
    else:
        lang = doc.lang
    doc: TextElement
    model = get_default_model(lang)
    sentences = doc.get_sentences()
    summary_size = int(len(sentences) * ratio) if word_count is None else 1
    assert 0 < summary_size < len(sentences)

    # get_doc_index = dict(zip(range(len(doc_indices)), doc_indices))
    similarity_matrix, clusters = process(sentences, model, k_ratio=k_ratio)

    # costs_ = {i: costs_dict[get_doc_index[i]] for i in range(tokenized_sentences_len)}
    summary_indices = greedy_submodular_maximization(similarity_matrix, clusters, None, summary_size, alpha, tradeoff_coefficient, word_count)

    return [sentences[idx] for idx in summary_indices]

