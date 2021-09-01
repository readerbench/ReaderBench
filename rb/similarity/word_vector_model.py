import os
from enum import Enum
from rb.similarity.vector_model import VectorModel, VectorModelType
from typing import Dict, List, Tuple, Union

import numpy as np
from numpy.linalg import norm
from rb.core.lang import Lang
from rb.core.text_element import TextElement
from rb.core.word import Word
from rb.similarity.vector import Vector
from rb.utils.downloader import check_version, download_model
from rb.utils.rblogger import Logger

WordSimilarity = Tuple[str, float]

logger = Logger.get_logger()

class CorporaEnum(Enum):
    README = 'readme'
    COCA = 'coca'
    JOSE_ANTONIO = 'jose_antonio'
    RNC_WIKIPEDIA = 'rnc_wikipedia'
    LE_MONDE = 'le_monde'
    WIKI = 'wiki'


class WordVectorModel(VectorModel):
    
    def __init__(self, type: VectorModelType, corpus: str, lang: Lang, size: int = 300, check_updates=True):
        VectorModel.__init__(self, type=type, name=f"{type.name}({corpus})", lang=lang, size=size)
        self.corpus = corpus
        self.vectors: Dict[str, Vector] = {}
        self.base_vectors: List[Vector] = []
        self.word_clusters: Dict[int, List[str]] = {}
        if check_updates and check_version(lang, self.corpus):
            if not download_model(lang, self.corpus):
                raise FileNotFoundError("Requested model ({}) not found for {}".format(self.corpus, lang.value))
        logger.info('Loading vectors.. ')
        self.load_vectors()
        logger.info('Vectors loaded')
        if len(self.vectors) > 100000:
            try:
                logger.info('Loading clusters.. ')
                self.load_clusters()
                logger.info('Clusters loaded')
            except:
                logger.info('Clusters not found! Building clusters.. ')
                self.build_clusters(8)
                self.save_clusters()
                logger.info('Clusters saved')
    
    def load_vectors(self):
        self.load_vectors_from_txt_file("resources/{}/models/{}/{}.model".format(self.lang.value, self.corpus, self.type.name))

    def get_vector(self, elem: Union[str, TextElement]) -> Vector:
        if isinstance(elem, str):
            return self.vectors[elem]
        if not self in elem.vectors:
            if isinstance(elem, Word):
                if elem.text in self.vectors:
                    elem.vectors[self] = self.vectors[elem.text]
                elif elem.lemma in self.vectors:
                    elem.vectors[self] = self.vectors[elem.lemma]
                else:
                    return None
            else:
                vectors = [self.get_vector(child) for child in elem.components]
                vectors = [v.values for v in vectors if v is not None]
                elem.vectors[self] = Vector(np.sum(vectors, axis=0)) if len(vectors) > 0 else None
        return elem.vectors[self]
    
    def similarity(self, a: Union[TextElement, Vector], b: Union[TextElement, Vector]) -> float:
        if isinstance(a, TextElement) and isinstance(b, TextElement) and a == b:
            return 1.0
        if isinstance(a, TextElement):
            a = self.get_vector(a)
        if isinstance(b, TextElement):
            b = self.get_vector(b)
        if a is None or b is None:
            return 0.0
        return Vector.cosine_similarity(a, b)

    """
    Loads vectors from file.
    If this method is used the self.size will be overwriten.
    """
    def load_vectors_from_txt_file(self, filepath: str) -> None:
        with open(filepath, 'r', encoding='utf-8') as f:
            line = f.readline()
            line_split = line.split()
            no_of_words = int(line_split[0])
            no_of_dimensions = int(line_split[1])
            self.size = no_of_dimensions

            for line in f:
                line_split = line.split()
                word = line_split[0]
                self.vectors[word] = Vector(np.array(line_split[1:], dtype=np.float))
            
    def compute_hash(self, v: Vector) -> int:
        result = 0
        for base in self.base_vectors:
            # print('base={}'.format(base))
            # print('values={}'.format(v.values))
            # print('result={}'.format(result))
            # print('base.shape={},len(v.values)={}'.format(len(base), len(v.values)))
            # print('np.dot={}'.format(np.dot(base, v.values)))
            # print('int={}'.format(int(np.dot(base, v.values) >= 0)))
            # print('result={}'.format((result << 1) + int(np.dot(base, v.values) >= 0)))
            result = (result << 1) + int(np.dot(base.values, v.values) >= 0)
        return result

    def build_clusters(self, n: int = 12):
        self.base_vectors = [Vector(v) for v in np.random.normal(size=(n, self.size))]
        self.word_clusters = {}
        for w, v in self.vectors.items():
            hash = self.compute_hash(v)
            if hash not in self.word_clusters:
                self.word_clusters[hash] = []
            self.word_clusters[hash].append(w) 

    def save_clusters(self):
        folder = f"resources/{self.lang.value}/models/{self.corpus}"
        os.makedirs(folder, exist_ok=True)     
    
        with open(f"{folder}/{self.type.name}-clusters.txt", "wt", encoding='utf-8') as f:
            f.write("{}\n".format(len(self.base_vectors)))
            for base in self.base_vectors:
                f.write(" ".join(str(x) for x in base.values) + "\n")
            f.write("{}\n".format(len(self.vectors)))
            for hash, words in self.word_clusters.items():
                for word in words:
                    f.write("{} {}\n".format(word, hash))

    def load_clusters(self):
        with open(f"resources/{self.lang.value}/models/{self.corpus}/{self.type.name}-clusters.txt", "rt", encoding="utf-8") as f:
            n = int(f.readline())
            for i in range(n):
                line = f.readline()
                self.base_vectors.append(Vector(np.array([float(x) for x in line.split(" ")])))
            n = int(f.readline())
            for i in range(n):
                line = f.readline()
                word, hash = line.split(" ")
                hash = int(hash)
                if hash not in self.word_clusters:
                    self.word_clusters[hash] = []
                self.word_clusters[hash].append(word) 

    def get_cluster(self, hash: int, vector: Vector, threshold: float = None) -> List[WordSimilarity]:
        if len(self.base_vectors) == 0:
            result = self.vectors.keys()
        else:
            result = self.word_clusters[hash] if hash in self.word_clusters else []
        result = [(word, self.similarity(self.get_vector(word), vector)) for word in result]
        if threshold is None:
            return result
        else:
            return [(word, sim) for word, sim in result if sim > threshold]

    def most_similar(self, elem: Union[str, TextElement, Vector], 
                    topN: int = 10, threshold: float = None) -> List[WordSimilarity]:
        if not isinstance(elem, type(Vector)):
            elem = self.get_vector(elem)
        if elem is None:
            return []
        hash = self.compute_hash(elem)
        cluster = self.get_cluster(hash, elem, threshold)
        if len(cluster) < topN:
            for i in range(len(self.base_vectors) - 1):
                new_hash = hash ^ (1 << (len(self.base_vectors) - i - 1))
                for j in range(i+1, len(self.base_vectors)):
                    new_hash = new_hash ^ (1 << (len(self.base_vectors) - j - 1))
                    cluster = cluster + self.get_cluster(new_hash, elem, threshold)
        return sorted(cluster, key=lambda x: x[1], reverse=True)[:topN]


