import numpy as np
import random
import math
import time
from collections import Counter


class StringKernel():

    def __init__(self):
        pass
    @staticmethod
    def compute_kernel_two_strings(string1, string2, ngram_range_min, ngram_range_max, clusters=None):
        pass

    @staticmethod
    def compute_kernel_string_listofstrings(string1, strings, ngram_range_min, ngram_range_max, clusters=None):
        pass

    @staticmethod
    def compute_kernel_listofstrings(strings, ngram_range_min, ngram_range_max, normalize=False):
        pass

    @staticmethod
    def run(dataset, ngram_range_min, ngram_range_max, normalize=False, clusters=None):
        pass

class IntersectionStringKernel(StringKernel):

    def __init__(self):
        super().__init__()

    @staticmethod
    def compute_kernel_two_strings(string1, string2, ngram_range_min, ngram_range_max, clusters=None):

        if clusters != None and len(clusters) > 0:
            return IntersectionStringKernel.compute_kernel_two_strings_clusters2(string1, string2, ngram_range_min, ngram_range_max, clusters)

        ngrams = {}
        for char_index, char in enumerate(string1):
            for d in range(ngram_range_min, ngram_range_max+1):
                if char_index + d <= len(string1):
                    ngram = string1[char_index:char_index+d]
                    if ngram not in ngrams:
                        ngrams[ngram] = 1
                    else:
                        ngrams[ngram] = ngrams[ngram] + 1
        kernel = 0
        for char_index, char in enumerate(string2):
            for d in range(ngram_range_min, ngram_range_max+1):
                if char_index + d <= len(string2):
                    ngram = string2[char_index:char_index+d]
                    if (ngram in ngrams) and (ngrams[ngram] > 0):
                        kernel += 1
                        ngrams[ngram] = ngrams[ngram] - 1
        return 1.0*kernel

    @staticmethod
    def compute_kernel_two_strings_clusters(string1, string2, ngram_range_min, ngram_range_max, clusters):

        ngrams = {}
        ngrams_in_clusters = {}

        for char_index, char in enumerate(string1):
            for d in range(ngram_range_min, ngram_range_max + 1):
                if char_index + d <= len(string1):
                    ngram = string1[char_index:char_index + d]
                    if ngram not in ngrams_in_clusters:
                        found = False
                        for cluster in clusters:
                            if ngram in cluster:
                                found = True
                                break
                        ngrams_in_clusters[ngram] = found
                    if ngrams_in_clusters[ngram] == False:
                        ngrams_in_clusters[ngram] = False
                        if ngram not in ngrams:
                            ngrams[ngram] = 1
                        else:
                            ngrams[ngram] = ngrams[ngram] + 1


        kernel = 0
        for char_index, char in enumerate(string2):
            for d in range(ngram_range_min, ngram_range_max + 1):
                if char_index + d <= len(string2):
                    ngram = string2[char_index:char_index + d]
                    if (ngram in ngrams) and (ngrams[ngram] > 0):
                        kernel += 1
                        ngrams[ngram] = ngrams[ngram] - 1

        d1 = {}
        d2 = {}

        for cl, _ in enumerate(clusters):
            d1[cl] = 0
            d2[cl] = 0

        for cl_index, cl in enumerate(clusters):
            for ngram in cl:
                if len(ngram) >= ngram_range_min and len(ngram) <= ngram_range_max:
                    d1[cl_index] = d1[cl_index] + string1.count(ngram)
                    d2[cl_index] = d2[cl_index] + string2.count(ngram)

        for cl, _ in enumerate(clusters):
            kernel += min(d1[cl],d2[cl])
        return kernel

    @staticmethod
    def compute_kernel_two_strings_clusters2(string1, string2, ngram_range_min, ngram_range_max, clusters):

        index = 0
        clusters_dict = {}
        for cluster in clusters:
            for ngram in cluster:
                clusters_dict[ngram] = index
            index += 1

        s1_ngram = Counter()
        s2_ngram = Counter()

        for char_index, char in enumerate(string1):
            for d in range(ngram_range_min, ngram_range_max + 1):
                if char_index + d <= len(string1):
                    ngram = string1[char_index:char_index + d]
                    if ngram not in clusters_dict:
                        clusters_dict[ngram] = index
                        index += 1
                    s1_ngram[clusters_dict[ngram]] += 1

        for char_index, char in enumerate(string2):
            for d in range(ngram_range_min, ngram_range_max + 1):
                if char_index + d <= len(string2):
                    ngram = string2[char_index:char_index + d]
                    if ngram not in clusters_dict:
                        clusters_dict[ngram] = index
                        index += 1
                    s2_ngram[clusters_dict[ngram]] += 1

        kernel = 0
        for c in s1_ngram:
            kernel += min(s1_ngram[c], s2_ngram[c])

        return kernel

    @staticmethod
    def compute_kernel_string_listofstrings(string1, strings, ngram_range_min, ngram_range_max, normalize=False, clusters=None):
        kernels = []
        for string2 in strings:
            pr = IntersectionStringKernel.compute_kernel_two_strings(string1, string2, ngram_range_min, ngram_range_max, clusters)
            if normalize == True:
                i = IntersectionStringKernel.compute_kernel_two_strings(string1, string1, ngram_range_min, ngram_range_max, clusters)
                j = IntersectionStringKernel.compute_kernel_two_strings(string2, string2, ngram_range_min, ngram_range_max, clusters)
                if i == 0.0 or j == 0.0:
                    kernels.append(0.0)
                else:
                    kernels.append(pr / math.sqrt(i*j))
            elif normalize == False:
                kernels.append(pr)

        kernels = np.array(kernels)
        return kernels

    @staticmethod
    def compute_kernel_listofstrings(strings, ngram_range_min, ngram_range_max, normalize=False):
        kernels = []
        for string1 in strings:
            kernels.append(IntersectionStringKernel.compute_kernel_string_listofstrings(string1, strings, ngram_range_min, ngram_range_max))

        if normalize == True:
            for i, aux in enumerate(kernels):
                for j, _ in enumerate(aux):
                    print (i,j, kernels[i][j], kernels[i][i], kernels[j][j], math.sqrt(kernels[i][i] * kernels[j][j]))
                    kernels[i][j] = kernels[i][j] /  math.sqrt(kernels[i][i] * kernels[j][j])
                    print (kernels[i][j])

        kernels = np.array(kernels)
        return kernels

    @staticmethod
    def run(dataset, ngram_range_min, ngram_range_max, normalize=False, clusters=None):
        correct = 0
        total = 0

        for entry_index, entry in enumerate(dataset.data):
            if len(entry.answer_pool) == 0:
                total += 1
                continue
            question = entry.question
            similarity_scores = IntersectionStringKernel.compute_kernel_string_listofstrings(question, entry.answer_pool, ngram_range_min, ngram_range_max, normalize, clusters)
            max_indexes = np.argwhere(similarity_scores == np.max(similarity_scores)).flatten().tolist()
            random_max_index = random.choice(max_indexes)
            if entry.answer_pool[random_max_index] in entry.correct_answer:
                correct += 1
            total += 1

        # print ("Intersection kernel (", ngram_range_min, ngram_range_max, normalize, ") =", 1.0* correct / total, flush=True)
        return 1.0 * correct / total

class PresenceStringKernel(StringKernel):

    def __init__(self):
        super().__init__()

    @staticmethod
    def compute_kernel_two_strings(string1, string2, ngram_range_min, ngram_range_max, clusters=None):
        if clusters != None and len(clusters) > 0:
            return PresenceStringKernel.compute_kernel_two_strings_clusters2(string1, string2, ngram_range_min, ngram_range_max, clusters)

        ngrams = {}
        for char_index, char in enumerate(string1):
            for d in range(ngram_range_min, ngram_range_max+1):
                if char_index + d <= len(string1):
                    ngram = string1[char_index:char_index+d]
                    ngrams[ngram] = 1
        kernel = 0
        for char_index, char in enumerate(string2):
            for d in range(ngram_range_min, ngram_range_max+1):
                if char_index + d <= len(string2):
                    ngram = string2[char_index:char_index+d]
                    if (ngram in ngrams):
                        kernel += 1
                        ngrams.pop(ngram)
        return 1.0*kernel

    @staticmethod
    def compute_kernel_two_strings_clusters(string1, string2, ngram_range_min, ngram_range_max, clusters):

        ngrams = {}
        ngrams_in_clusters = {}

        for char_index, char in enumerate(string1):
            for d in range(ngram_range_min, ngram_range_max + 1):
                if char_index + d <= len(string1):
                    ngram = string1[char_index:char_index + d]
                    if ngram not in ngrams_in_clusters:
                        found = False
                        for cluster in clusters:
                            if ngram in cluster:
                                found = True
                                break
                        ngrams_in_clusters[ngram] = found
                    if ngrams_in_clusters[ngram] == False:
                        if ngram not in ngrams:
                            ngrams[ngram] = 1
                        else:
                            ngrams[ngram] = ngrams[ngram] + 1

        kernel = 0
        for char_index, char in enumerate(string2):
            for d in range(ngram_range_min, ngram_range_max + 1):
                if char_index + d <= len(string2):
                    ngram = string2[char_index:char_index + d]
                    if (ngram in ngrams):
                        kernel += 1
                        ngrams.pop(ngram)

        d1 = {}
        d2 = {}

        for cl, _ in enumerate(clusters):
            d1[cl] = 0
            d2[cl] = 0

        for cl_index, cl in enumerate(clusters):
            for ngram in cl:
                if len(ngram) >= ngram_range_min and len(ngram) <= ngram_range_max:
                    d1[cl_index] = d1[cl_index] + string1.count(ngram)
                    d2[cl_index] = d2[cl_index] + string2.count(ngram)

        for cl, _ in enumerate(clusters):
            if d1[cl] * d2[cl] == 0:
                continue
            kernel += 1
        return kernel

    @staticmethod
    def compute_kernel_two_strings_clusters2(string1, string2, ngram_range_min, ngram_range_max, clusters):

        index = 0
        clusters_dict = {}
        for cluster in clusters:
            for ngram in cluster:
                clusters_dict[ngram] = index
            index += 1

        s1_ngram = Counter()
        s2_ngram = Counter()

        for char_index, char in enumerate(string1):
            for d in range(ngram_range_min, ngram_range_max + 1):
                if char_index + d <= len(string1):
                    ngram = string1[char_index:char_index + d]
                    if ngram not in clusters_dict:
                        clusters_dict[ngram] = index
                        index += 1
                    s1_ngram[clusters_dict[ngram]] += 1

        for char_index, char in enumerate(string2):
            for d in range(ngram_range_min, ngram_range_max + 1):
                if char_index + d <= len(string2):
                    ngram = string2[char_index:char_index + d]
                    if ngram not in clusters_dict:
                        clusters_dict[ngram] = index
                        index += 1
                    s2_ngram[clusters_dict[ngram]] += 1

        kernel = 0
        for c in s1_ngram:
            if s1_ngram[c] * s2_ngram[c] >= 1:
                kernel += 1
        return kernel

    @staticmethod
    def compute_kernel_string_listofstrings(string1, strings, ngram_range_min, ngram_range_max, normalize=False, clusters=None):
        kernels = []
        for string2 in strings:
            pr = PresenceStringKernel.compute_kernel_two_strings(string1, string2, ngram_range_min, ngram_range_max, clusters)
            if normalize == True:
                i = PresenceStringKernel.compute_kernel_two_strings(string1, string1, ngram_range_min, ngram_range_max, clusters)
                j = PresenceStringKernel.compute_kernel_two_strings(string2, string2, ngram_range_min, ngram_range_max, clusters)
                if i == 0.0 or j == 0.0:
                    kernels.append(0.0)
                else:
                    kernels.append(pr / math.sqrt(i*j))
            elif normalize == False:
                kernels.append(pr)

        kernels = np.array(kernels)
        return kernels

    @staticmethod
    def compute_kernel_listofstrings(strings, ngram_range_min, ngram_range_max, normalize=False):
        kernels = []
        for string1 in strings:
            kernels.append(PresenceStringKernel.compute_kernel_string_listofstrings(string1, strings, ngram_range_min, ngram_range_max))

        if normalize == True:
            for i, aux in enumerate(kernels):
                for j, _ in enumerate(aux):
                    print (i,j, kernels[i][j], kernels[i][i], kernels[j][j], math.sqrt(kernels[i][i] * kernels[j][j]))
                    kernels[i][j] = kernels[i][j] /  math.sqrt(kernels[i][i] * kernels[j][j])
                    print (kernels[i][j])

        kernels = np.array(kernels)
        return kernels

    @staticmethod
    def run(dataset, ngram_range_min, ngram_range_max, normalize=False, clusters=None):
        correct = 0
        total = 0

        for entry_index, entry in enumerate(dataset.data):
            if len(entry.answer_pool) == 0:
                total += 1
                continue
            question = entry.question
            similarity_scores = PresenceStringKernel.compute_kernel_string_listofstrings(question, entry.answer_pool, ngram_range_min, ngram_range_max, normalize, clusters)
            max_indexes = np.argwhere(similarity_scores == np.max(similarity_scores)).flatten().tolist()
            random_max_index = random.choice(max_indexes)
            if entry.answer_pool[random_max_index] in entry.correct_answer:
                correct += 1
            total += 1

        # print("Presence kernel (", ngram_range_min, ngram_range_max, normalize, ") =", 1.0 * correct / total, flush=True)
        return 1.0* correct / total

class SpectrumStringKernel(StringKernel):

    def __init__(self):
        super().__init__()

    @staticmethod
    def compute_kernel_two_strings(string1, string2, ngram_range_min, ngram_range_max, clusters=None):

        if clusters != None and len(clusters) > 0:
            return SpectrumStringKernel.compute_kernel_two_strings_clusters2(string1, string2, ngram_range_min, ngram_range_max, clusters)

        ngrams = {}
        for char_index, char in enumerate(string1):
            for d in range(ngram_range_min, ngram_range_max+1):
                if char_index + d <= len(string1):
                    ngram = string1[char_index:char_index+d]
                    if ngram not in ngrams:
                        ngrams[ngram] = 1
                    else:
                        ngrams[ngram] = ngrams[ngram] + 1
        kernel = 0
        for char_index, char in enumerate(string2):
            for d in range(ngram_range_min, ngram_range_max+1):
                if char_index + d <= len(string2):
                    ngram = string2[char_index:char_index+d]
                    if (ngram in ngrams):
                        kernel += ngrams[ngram]

        return 1.0*kernel

    @staticmethod
    def compute_kernel_two_strings_clusters(string1, string2, ngram_range_min, ngram_range_max, clusters):

        ngrams = {}
        ngrams_in_clusters = {}

        for char_index, char in enumerate(string1):
            for d in range(ngram_range_min, ngram_range_max+1):
                if char_index + d <= len(string1):
                    ngram = string1[char_index:char_index+d]
                    if ngram not in ngrams_in_clusters:
                        found = False
                        for cluster in clusters:
                            if ngram in cluster:
                                found = True
                                break
                        ngrams_in_clusters[ngram] = found
                    if ngrams_in_clusters[ngram] == False:
                        ngrams_in_clusters[ngram] = False
                        if ngram not in ngrams:
                            ngrams[ngram] = 1
                        else:
                            ngrams[ngram] = ngrams[ngram] + 1

        kernel = 0
        for char_index, char in enumerate(string2):
            for d in range(ngram_range_min, ngram_range_max+1):
                if char_index + d <= len(string2):
                    ngram = string2[char_index:char_index+d]
                    if (ngram in ngrams):
                        kernel += ngrams[ngram]

        d1 = {}
        d2 = {}

        for cl, _ in enumerate(clusters):
            d1[cl] = 0
            d2[cl] = 0

        for cl_index, cl in enumerate(clusters):
            for ngram in cl:
                if len(ngram) >= ngram_range_min and len(ngram) <= ngram_range_max:
                    d1[cl_index] = d1[cl_index] + string1.count(ngram)
                    d2[cl_index] = d2[cl_index] + string2.count(ngram)

        for cl, _ in enumerate(clusters):
            kernel += d1[cl] * d2[cl]
        return kernel

    @staticmethod
    def compute_kernel_two_strings_clusters2(string1, string2, ngram_range_min, ngram_range_max, clusters):

        index = 0
        clusters_dict = {}
        for cluster in clusters:
            for ngram in cluster:
                clusters_dict[ngram] = index
            index += 1

        s1_ngram = Counter()
        s2_ngram = Counter()

        for char_index, char in enumerate(string1):
            for d in range(ngram_range_min, ngram_range_max + 1):
                if char_index + d <= len(string1):
                    ngram = string1[char_index:char_index + d]
                    if ngram not in clusters_dict:
                        clusters_dict[ngram] = index
                        index += 1
                    s1_ngram[clusters_dict[ngram]] += 1

        for char_index, char in enumerate(string2):
            for d in range(ngram_range_min, ngram_range_max + 1):
                if char_index + d <= len(string2):
                    ngram = string2[char_index:char_index + d]
                    if ngram not in clusters_dict:
                        clusters_dict[ngram] = index
                        index += 1
                    s2_ngram[clusters_dict[ngram]] += 1

        kernel = 0
        for c in s1_ngram:
            # print (c, s1_ngram[c], s2_ngram[c])
            kernel += s1_ngram[c] * s2_ngram[c]
        return kernel

    @staticmethod
    def compute_kernel_string_listofstrings(string1, strings, ngram_range_min, ngram_range_max, normalize=False, clusters=None):
        kernels = []
        for string2 in strings:
            pr = SpectrumStringKernel.compute_kernel_two_strings(string1, string2, ngram_range_min, ngram_range_max, clusters)
            if normalize == True:
                i = SpectrumStringKernel.compute_kernel_two_strings(string1, string1, ngram_range_min, ngram_range_max, clusters)
                j = SpectrumStringKernel.compute_kernel_two_strings(string2, string2, ngram_range_min, ngram_range_max, clusters)
                if i == 0.0 or j == 0.0:
                    kernels.append(0.0)
                else:
                    kernels.append(pr / math.sqrt(i*j))
            elif normalize == False:
                kernels.append(pr)

        kernels = np.array(kernels)
        return kernels

    @staticmethod
    def compute_kernel_listofstrings(strings, ngram_range_min, ngram_range_max, normalize=False):
        kernels = []
        for string1 in strings:
            kernels.append(SpectrumStringKernel.compute_kernel_string_listofstrings(string1, strings, ngram_range_min, ngram_range_max))

        if normalize == True:
            for i, aux in enumerate(kernels):
                for j, _ in enumerate(aux):
                    print (i,j, kernels[i][j], kernels[i][i], kernels[j][j], math.sqrt(kernels[i][i] * kernels[j][j]))
                    kernels[i][j] = kernels[i][j] /  math.sqrt(kernels[i][i] * kernels[j][j])
                    print (kernels[i][j])

        kernels = np.array(kernels)
        return kernels

    @staticmethod
    def run(dataset, ngram_range_min, ngram_range_max, normalize=False, clusters=None):
        correct = 0
        total = 0

        for entry_index, entry in enumerate(dataset.data):
            if len(entry.answer_pool) == 0:
                total += 1
                continue
            question = entry.question
            similarity_scores = SpectrumStringKernel.compute_kernel_string_listofstrings(question, entry.answer_pool, ngram_range_min, ngram_range_max, normalize, clusters)
            max_indexes = np.argwhere(similarity_scores == np.max(similarity_scores)).flatten().tolist()
            random_max_index = random.choice(max_indexes)
            if entry.answer_pool[random_max_index] in entry.correct_answer:
                correct += 1
            total += 1

        # print("Spectrum kernel (", ngram_range_min, ngram_range_max, normalize, ") =", 1.0 * correct / total, flush=True)
        return 1.0 * correct / total


if __name__ == "__main__":

    a = "walking kin lorem ipsum dem dolort adsr su u  how are you to come to me to go to stay some things we want to change but no change vauce forte nefaster pentru a recladi nimeni nu ma munca"
    b = ["de vreo miscare de rezinta hufhre stii cum un singur patid o singura germanie wtf is this i am wathcing ot stay stalin wtf is do you want to thisn"]

    t = time.time()
    for _ in range(100):
        c = IntersectionStringKernel.compute_kernel_string_listofstrings(a, b, 1, 10, normalize=False)
    print (c, time.time()-t)

    t = time.time()
    for _ in range(100):
        c = IntersectionStringKernel.compute_kernel_string_listofstrings(a, b, 1, 10, normalize=False, clusters=[["aaaa"],["bbbb"]])
    print(c , time.time() - t)

