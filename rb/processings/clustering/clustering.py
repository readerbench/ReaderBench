import csv
import os
import uuid
from typing import Dict, List, Tuple

import numpy as np
import time, warnings
from matplotlib import pyplot as plt
from scipy import ndimage
from sklearn.base import BaseEstimator, clone
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.metaestimators import if_delegate_has_method
from sklearn.preprocessing import StandardScaler
from sklearn import cluster
from itertools import cycle, islice
from rb.cna.cna_graph import CnaGraph
from rb.complexity.complexity_index import ComplexityIndex, compute_indices
from rb.core.document import Document
from rb.core.lang import Lang
from rb.core.text_element_type import TextElementType
from rb.similarity.vector_model import (CorporaEnum, VectorModel,
                                        VectorModelType)
from rb.similarity.vector_model_factory import (VECTOR_MODELS,
                                                create_vector_model)
from rb.utils.rblogger import Logger

logger = Logger.get_logger()

class Clustering:

    def __init__(self):
        pass

    def plot_clustering(self, X_red, labels, title=None):
        x_min, x_max = np.min(X_red, axis=0), np.max(X_red, axis=0)
        X_red = (X_red - x_min) / (x_max - x_min)
        plt.figure(figsize=(6, 4))
        for i in range(X_red.shape[0]):
            plt.text(X_red[i, 0], X_red[i, 1], str(self.y[i]),
                    color=plt.cm.nipy_spectral(labels[i] / 10.),
                    fontdict={'weight': 'bold', 'size': 9})

        plt.xticks([])
        plt.yticks([])
        if title is not None:
            plt.title(title, size=17)
        plt.axis('off')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    def plot_scatter(self, X,  color, alpha=0.5):
        return plt.scatter(X[:, 0],
                       X[:, 1],
                       c=color,
                       alpha=alpha,
                       edgecolor='k')

    def compute_clustering(self):
        categories = ['stats_general.csv', 'stats_literature.csv', 'stats_science.csv']
        values = {0: [], 1: [], 2: []}
        labels = []
        for i, cat in enumerate(categories):
            stats = csv.reader(open(os.path.join('categories_readme/new_stats/', cat),
                 'rt', encoding='utf-8'))
            for j, row in enumerate(stats):   
                if j == 0:  continue
                vs = []
                for v in row[1:]:
                    vs.append(float(v))
                values[i].append(vs)
                labels.append(i)
        
        gen, lit, science  = np.asarray(values[0]), np.asarray(values[1]), np.asarray(values[2])
        all_samples = np.concatenate((gen, lit, science), axis=0)
        labels = np.asarray(labels)
        y = labels
        X = all_samples

        # clusterer = AgglomerativeClustering(n_clusters=3)
        # cluster_labels = clusterer.fit_predict(X)

        # plt.figure(figsize=(12, 4))

        # plt.subplot(131)
        # self.plot_scatter(X, cluster_labels)
        # plt.title("Ward Linkage")
        # for linkage in ('ward', 'average', 'complete', 'single'):
        #     clustering = AgglomerativeClustering(linkage=linkage, n_clusters=3)
        #     clustering.fit(self.X)
        #     self.plot_clustering(self.X, clustering.labels_, "%s linkage" % linkage)
        # update parameters with dataset-specific values


        # normalize dataset for easier parameter selection
        X = StandardScaler().fit_transform(X)

        # ============
        # Create cluster objects
        # ============
        ward = cluster.AgglomerativeClustering(
            n_clusters=3, linkage='ward')
        complete = cluster.AgglomerativeClustering(
            n_clusters=3, linkage='complete')
        average = cluster.AgglomerativeClustering(
            n_clusters=3, linkage='average')
        single = cluster.AgglomerativeClustering(
            n_clusters=3, linkage='single')
        brc = cluster.Birch(n_clusters=3)

        clustering_algorithms = (
            ('Complete Linkage', complete),
            ('Ward Linkage', ward),
            ('Birch', brc)
        )
        plot_num = 1
        colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a']),
                                        int(max(y) + 1))))
        for name, algorithm in clustering_algorithms:
            t0 = time.time()

            # catch warnings related to kneighbors_graph
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="the number of connected components of the " +
                    "connectivity matrix is [0-9]{1,2}" +
                    " > 1. Completing it to avoid stopping the tree early.",
                    category=UserWarning)
                algorithm.fit(X)

            t1 = time.time()
            if hasattr(algorithm, 'labels_'):
                y_pred = algorithm.labels_.astype(np.int)
            else:
                y_pred = algorithm.predict(X)

            plt.subplot(1, len(clustering_algorithms) + 1, plot_num)
            plt.title(name, size=18)

            
            plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[y_pred])

            plt.xlim(-2.5, 2.5)
            plt.ylim(-2.5, 2.5)
            plt.xticks(())
            plt.yticks(())
            # plt.text(.99, .01, ('%.2fs' % (t1 - t0)).lstrip('0'),
            #         transform=plt.gca().transAxes, size=15,
            #         horizontalalignment='right')
            plot_num += 1

        plt.subplot(1, len(clustering_algorithms) + 1, plot_num)
        plt.title('True Classes', size=18)

       
        plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[y])

        plt.xlim(-2.5, 2.5)
        plt.ylim(-2.5, 2.5)
        plt.xticks(())
        plt.yticks(())
        # plt.text(.99, .01,
        #         transform=plt.gca().transAxes, size=15,
        #         horizontalalignment='right')
        plt.show()


       
