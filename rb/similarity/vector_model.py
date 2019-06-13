import os
from enum import Enum
from typing import Dict, List, Tuple, Union

import numpy as np
from numpy.linalg import norm
from rb.core.lang import Lang
from rb.core.text_element import TextElement
from rb.core.word import Word
from rb.utils.downloader import check_version, download_model
WordSimilarity = Tuple[str, float]



class VectorModelType(Enum):
    LSA = 0
    LDA = 1
    WORD2VEC = 2
    FASTTEXT = 3
    GLOVE = 4


class VectorModel:

    
    def __init__(self, type: VectorModelType, name: str, lang: Lang, size: int = 300):
        self.lang = lang
        self.type = type
        self.name = name
        self.size = size
        self.vectors: Dict[str, np.ndarray] = {}
        self.base_vectors: List[np.ndarray] = []
        self.word_clusters: Dict[int, List[str]] = {}
        corpus = "resources/{}/models/{}".format(lang.value, name)
        if check_version(lang, name):
            if not download_model(lang, name):
                raise FileNotFoundError("Requested model ({}) not found for {}".format(name, lang.value))
        self.load_vectors()
        if len(self.vectors) > 100000:
            try:
                self.load_clusters()
            except:
                self.build_clusters(8)
                self.save_clusters()
        

    def load_vectors(self):
        self.load_vectors_from_txt_file("resources/{}/models/{}/{}.model".format(self.lang.value, self.name, self.type.name))

    def get_vector(self, elem: Union[str, TextElement]) -> np.ndarray:
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
                vectors = [v for v in vectors if v is not None]
                elem.vectors[self] = np.sum(vectors, axis=0) if len(vectors) > 0 else None
        return elem.vectors[self]
            
    
    def similarity(self, a: Union[TextElement, np.ndarray], b: Union[TextElement, np.ndarray]) -> float:
        if isinstance(a, TextElement) and isinstance(b, TextElement) and a == b:
            return 1.0
        if isinstance(a, TextElement):
            a = self.get_vector(a)
        if isinstance(b, TextElement):
            b = self.get_vector(b)
        if a is None or b is None:
            return 0.0
        return np.dot(a, b) / (norm(a) * norm(b))


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

            for _ in range(no_of_words):
                line = f.readline()
                line_split = line.split()
                word = line_split[0]
                self.vectors[word] = np.array([float(x) for x in line_split[1:]])
            
    def compute_hash(self, v: np.ndarray) -> int:
        result = 0
        for base in self.base_vectors:
            result = (result << 1) + int(np.dot(base, v) >= 0)
        return result

    def build_clusters(self, n: int = 12):
        self.base_vectors = np.random.normal(size=(n, self.size)).tolist()
        self.word_clusters = {}
        for w, v in self.vectors.items():
            hash = self.compute_hash(v)
            if hash not in self.word_clusters:
                self.word_clusters[hash] = []
            self.word_clusters[hash].append(w) 

    def save_clusters(self):
        folder = "resources/{}/models/{}".format(self.lang.value, self.name)
        os.makedirs(folder, exist_ok=True)     
    
        with open("{}/{}-clusters.txt".format(folder, self.type.name), "wt") as f:
            f.write("{}\n".format(len(self.base_vectors)))
            for base in self.base_vectors:
                f.write(" ".join(str(x) for x in base) + "\n")
            f.write("{}\n".format(len(self.vectors)))
            for hash, words in self.word_clusters.items():
                for word in words:
                    f.write("{} {}\n".format(word, hash))

    def load_clusters(self):
        with open("resources/{}/models/{}/{}-clusters.txt".format(self.lang.value, self.name, self.type.name), "rt") as f:
            n = int(f.readline())
            for i in range(n):
                line = f.readline()
                self.base_vectors.append(np.array([float(x) for x in line.split(" ")]))
            n = int(f.readline())
            for i in range(n):
                line = f.readline()
                word, hash = line.split(" ")
                hash = int(hash)
                if hash not in self.word_clusters:
                    self.word_clusters[hash] = []
                self.word_clusters[hash].append(word) 

    def get_cluster(self, hash: int, vector: np.ndarray, threshold: float = None) -> List[WordSimilarity]:
        if len(self.base_vectors) == 0:
            result = self.vectors.keys()
        else:
            result = self.word_clusters[hash] if hash in self.word_clusters else []
        result = [(word, self.similarity(self.get_vector(word), vector)) for word in result]
        if threshold is None:
            return result
        else:
            return [(word, sim) for word, sim in result if sim > threshold]

    def most_similar(self, elem: Union[str, TextElement, np.ndarray], topN: int = 10, threshold: float = None) -> List[WordSimilarity]:
        if not isinstance(elem, type(np.ndarray)):
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
        
