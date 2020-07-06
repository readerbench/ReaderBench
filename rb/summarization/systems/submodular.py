import heapq
import math
import os

from rb.summarization.rouge import Rouge
from rb.summarization.systems.summarizer_abc import Summarizer
from rb.summarization.utils.utils import *

from collections import defaultdict
from functools import reduce
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, Iterable

from rblogger import Logger

logger = Logger.get_logger()
logger.propagate = False

use_word_embeddings = False


class Submodular(Summarizer):
    
    def __init__(self):
        Summarizer.__init__(self)
        self._alpha_constant = 4
        self._k_ratio = 0.1
        self._tradeoff_coefficient = 5
    
    @property
    def alpha_constant(self):
        return self._alpha_constant
    
    @alpha_constant.setter
    def alpha_constant(self, value):
        self._alpha_constant = value

    @property
    def k_ratio(self):
        return self._k_ratio
    
    @k_ratio.setter
    def k_ratio(self, value):
        self._k_ratio = value
    
    @property
    def tradeoff_coefficient(self):
        return self._tradeoff_coefficient
    
    @tradeoff_coefficient.setter
    def tradeoff_coefficient(self, value):
        self._tradeoff_coefficient = value

    @staticmethod
    def process(tokenized_sentences, k_ratio: float = 0.2):
        """
        :param tokenized_sentences: every sentence is represented as a list of tokens (strings or word embeddings)
        :param k_ratio: represents the ratio between the number of clusters and the number of docs
        :return: tuple(similarity matrix, dictionary of clusters)
        """
        
        tokenized_sentences = list(tokenized_sentences)
        # for sent in tokenized_sentences:
        #     print(sent)
        no_sentences = len(tokenized_sentences)
        global use_word_embeddings
        if use_word_embeddings:
            """
            feature_matrix = np.zeros(shape=(no_sentences, 300))
            for i, doc in enumerate(tokenized_sentences):
                vector = np.zeros(300)
                for token in doc:
                    vector = vector + token.vector
                feature_matrix[i] = np.divide(vector, len(doc))
            similarity_matrix = combined_similarity(tokenized_sentences)
            """
            similarity_matrix = np.zeros([no_sentences, no_sentences])
            for i, j in itertools.combinations(range(no_sentences), 2):
                wmd_dist = tokenized_sentences[i].similarity(tokenized_sentences[j])
                similarity_matrix[i, j] = similarity_matrix[j, i] = 1.0 / wmd_dist if wmd_dist != 0 else wmd_dist
        else:
            sentences = [' '.join(tokens) for tokens in tokenized_sentences]
            vec, feature_matrix = build_feature_matrix(docs=sentences,
                                                       feature_type=FeatureType.TFIDF,
                                                       ngram_range=(1, 2),
                                                       min_df=0.1,
                                                       max_df=0.9)
            similarity_matrix = cosine_similarity(feature_matrix)

        # return similarity_matrix, {}

        assert (0 < k_ratio <= 1), "A value between 0 and 1 must be provided for k_ratio"
        no_clusters = math.ceil(k_ratio * no_sentences)
        km_obj, clusters = k_means(feature_matrix=feature_matrix, no_clusters=no_clusters, max_iter=20)
        """
        Note: 'clusters' is a list of size n which has values from 0 to k-1 and maps every
        sentence index to the belonging cluster index
        e.g. clusters=[1 1 0 1 1 0 1 0 1] for k=2
        """
        clusters_dict = {k: get_indices(clusters, k) for k in range(no_clusters)}

        return similarity_matrix, clusters_dict

    @staticmethod
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

    @staticmethod
    def diversity_reward_function(similarity_matrix, clusters: Dict, summary_indices: List) -> float:
        diversity_reward = 0.0
        if len(summary_indices) == 0:
            return diversity_reward

        no_clusters = len(clusters)
        for k in range(no_clusters):
            common_indices = list_intersection(summary_indices, clusters[k])
            if len(common_indices) == 0:
                continue
            cluster_reward = sum(map(lambda lst: sum(lst) / len(lst), [similarity_matrix[j] for j in common_indices]))
            diversity_reward += math.sqrt(cluster_reward)
        
        return diversity_reward
    
    def greedy_submodular_maximization(self, similarity_matrix, clusters, costs_dict, summary_size, word_count=None):
        n = len(similarity_matrix)
        prev_score = 0

        G = set()
        U = list(range(n))
        # print("alpha %d ratio %f trade-off %d" % (self.alpha_constant, self.k_ratio, self.tradeoff_coefficient))

        while len(U) > 0:
            temp_scores = []
            for l in U:
                # c_l = costs_dict[l]
                temp_summary_indices = list(G) + [l]

                coverage_result = self.coverage_function(similarity_matrix, temp_summary_indices, self.alpha_constant)
                diversity_reward = self.diversity_reward_function(similarity_matrix, clusters, temp_summary_indices)
                current_score = coverage_result + self.tradeoff_coefficient * diversity_reward

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

        """
        best_score = prev_score
        best_singleton_index = None
        for v in range(n):
            if costs_dict[v] <= word_count:
                coverage_result = self.coverage_function(similarity_matrix, [v], self.alpha_constant)
                diversity_reward = self.diversity_reward_function(similarity_matrix, clusters, [v])
                singleton_score = coverage_result + self.tradeoff_coefficient * diversity_reward
                if singleton_score > best_score:
                    best_score = singleton_score
                    best_singleton_index = v

        if best_singleton_index:
            return [best_singleton_index]
        """
        return sorted(G)

        # summary_indices = []
        # for i in range(summary_size):
        #     temp_results = []
        #     for j in range(n):
        #         if j in summary_indices:
        #             continue
        #         temp_summary_indices = summary_indices + [j]
        #
        #         coverage_result = self.coverage_function(similarity_matrix, temp_summary_indices, self.alpha_constant)
        #         diversity_reward = self.diversity_reward_function(similarity_matrix, clusters, temp_summary_indices)
        #
        #         score = coverage_result + self.tradeoff_coefficient * diversity_reward
        #         # exploit heapq properties by pushing the negative score (positive max -> negative min)
        #         heapq.heappush(temp_results, (-score, j))
        #
        #     # heappop(heap) :- This function is used to remove and return the smallest element from heap.
        #     heapq.heappush(summary_indices, heapq.heappop(temp_results)[1])     # get only the index
        #
        # # return the indices in ascending order
        # return [heapq.heappop(summary_indices) for _ in range(summary_size)]

    def summarize(self, doc, lang=Lang.EN, parser=None, ratio=0.2, word_count=None) -> Iterable[str]:
        if not parser:
            parser = self.parser

        # 1. split text into sententences
        doc_sentences = parser.tokenize_sentences(doc)
        # 2. - strip tags
        #    - strip multiple whitespaces
        #    - strip leading punctuation
        doc_sentences = list(map(lambda s: parser.preprocess(s, lang.value), doc_sentences))
        # costs_dict = {i: len(sentence.split()) for i, sentence in enumerate(doc_sentences)}

        # 3. tokenize sentences
        global use_word_embeddings
        if use_word_embeddings:
            doc_indices, tokenized_sentences = zip(*tokenize2(parser, doc_sentences, lang))
        else:
            doc_indices, tokenized_sentences = zip(*tokenize(parser, doc_sentences, lang))

        # summary_size = math.ceil(len(doc_sentences) * ratio) if word_count is None else 1
        summary_size = int(len(doc_sentences) * ratio) if word_count is None else 1
        tokenized_sentences_len = len(tokenized_sentences)
        assert 0 < summary_size < tokenized_sentences_len

        get_doc_index = dict(zip(range(len(doc_indices)), doc_indices))
        similarity_matrix, clusters = self.process(tokenized_sentences, k_ratio=self.k_ratio)

        # costs_ = {i: costs_dict[get_doc_index[i]] for i in range(tokenized_sentences_len)}
        summary_indices = self.greedy_submodular_maximization(similarity_matrix, clusters, None, summary_size, word_count)

        # for i in summary_indices:
        #     yield doc_sentences[get_doc_index[i]]

        summary_sentences = map(lambda i: doc_sentences[get_doc_index[i]], summary_indices)
        # return {'summary': ' '.join(summary_sentences)}
        return list(summary_sentences)


def grid_search():
    submodular_summarizer = Submodular()
    relative_path_rouge = Path.cwd() / ".." / "rouge"

    alpha_constant_candidates = [4]
    k_ratio_candidates = [0.1, 0.2]
    tradeoff_coeficient_candidates = range(5, 8)

    iteration = 0
    for a in alpha_constant_candidates:
        for b in k_ratio_candidates:
            for c in tradeoff_coeficient_candidates:
                header = "Round=%d a=%d b=%f c=%d" % (iteration, a, b, c)
                print(header)
                submodular_summarizer.alpha_constant = a
                submodular_summarizer.k_ratio = b
                submodular_summarizer.tradeoff_coefficient = c

                # summarize_dataset(dataset_type=DatasetType.DUC_2001, summarizer=submodular_summarizer)

                rouge = Rouge(relative_path_rouge)
                try:
                    res = rouge.evaluate()
                    # output_formatted = "round: %d alpha: %f k_ratio: %f trade-off: %d rouge_1_recall: %f rouge_1_precision: %f rouge_1_f_score: %f" % (
                    #                     iteration, a, b, c, res["rouge_1_recall"], res["rouge_1_precision"], res["rouge_1_f_score"])
                    output_formatted = "round: %d alpha: %f k_ratio: %f trade-off: %d" % (iteration, a, b, c)
                    result_filename = "duc2001.results.txt"
                    append_list_to_file(Path.cwd() / ".." / result_filename, [output_formatted, res])
                except Exception as inst:
                    logger.exception(type(inst), inst.args, inst)
                finally:
                    os.system("rm -rf %s/*" % str(relative_path_rouge / "model_summaries"))
                    os.system("rm -rf %s/*" % str(relative_path_rouge / "system_summaries"))

                iteration = iteration + 1


def main():
    submodular_summarizer = Submodular()

    en_doc = """
            Elephants are large mammals of the family Elephantidae
            and the order Proboscidea. Two species are traditionally recognised,
            the African elephant and the Asian elephant. Elephants are scattered
            throughout sub-Saharan Africa, South Asia, and Southeast Asia. Male
            African elephants are the largest extant terrestrial animals. All
            elephants have a long trunk used for many purposes,
            particularly breathing, lifting water and grasping objects. Their
            incisors grow into tusks, which can serve as weapons and as tools
            for moving objects and digging. Elephants' large ear flaps help
            to control their body temperature. Their pillar-like legs can
            carry their great weight. African elephants have larger ears
            and concave backs while Asian elephants have smaller ears
            and convex or level backs.  
        """

    ro_doc = """
            Ca orice roman, Ion este o specie a genului epic, în proză, de mare întindere, cu acţiune complexă
            desfăşurată pe mai multe planuri narative, organizate prin alternanţă sau înlănţuire, cu o intrigă
            amplă şi complicată. Personajele numeroase, de diverse tipologii dar bine individualizate, sunt
            angrenate în conflicte puternice, iar structura narativă realistă profilează o imagine consistentă
            şi profundă a vieţii. Principalul mod de expunere este naraţiunea, iar personajele se conturează
            direct prin descriere şi indirect, din propriile fapte, gânduri şi vorbe, cu ajutorul dialogului,
            al monologului interior şi al introspecţiei auctoriale.
        """

    ro_doc2 = """
            Elefanții sunt mamifere mari care formează familia Elephantidae din 
            ordinul Proboscidea. În prezent sunt cunoscute două specii, elefantul 
            african și elefantul asiatic. Elefanții africani masculi sunt cele mai 
            mari animale terestre. Toți elefanții au câteva trăsături distinctive, 
            dintre care cel mai notabil este trompa lungă utilizată în multe scopuri, 
            în special respirație, ridicarea apei și aruncarea obiectelor.
        """

    big_doc = """
           Even the scientists came to cross-purposes in designing this
        little world. Botanists wanted wind for pollination. Entomologists
        did not want their bugs sucked up by big fans. The people in the
        middle were the engineers, constantly revising plans, compromising.
           It is the professional nature of engineers and architects to
        keep the outside world out. Now suddenly they had to devise ways to
        keep the wild world in. As for the scientists, they were used to
        having the whole outdoors as a laboratory. Now they have to be
        satisfied with a very confined indoors.
           Consider hummingbirds, great little pollinators. But a pair of
        hummingbirds need the nectar of 3,200 flowers a day for energy.
        Scientists had to be sure there would be that many in bloom every
        day of the year.
           At the outset, no one thought what would happen to the airtight
        bubble with the air heating up and expanding during the day. It
        would explode the glass. So the engineers devised external,
        inflatable lungs that would store the excess until it cooled for
        return to the biosphere.
           With a closed water supply, engineers also had to devise
        exterior tanks to cool the liquid. In short, this small world in
        some cases is too small for its own good.
           Nevertheless it will be the same air and water year-round. The
        oxygen is breathed in by humans and animals who breathe out carbon
        dioxide which is breathed in by the plants and used in
        photosynthesis which produces more oxygen.
           Biosphere II is built on a hill to take advantage of the natural
        proclivity of water. A stream begins on an 85-foot mountain that
        dominates the rain forest. It then flows through the plains and
        grasslands of the savanna, down to the 25-foot deep ocean with its
        coral reef and then to the salt and freshwater marshes, drying up
        before it reaches the desert.
           The water-laden air then is drawn and climbs the mountain where
        natural cooling and condensing coils bring it down as rain and dew
        to feed the stream again.
           The desert required a compromise. There was no way to reduce the
        humidity. So Dr. Tony Burgess of the University of Arizona
        patterned the deserts after those of Baja California where the
        cactus and other plants have learned to live with ocean fog.
           Perhaps the most difficult design was Dr. Walter Addey's for the
        salt and freshwater marshes. He built prototypes, one in the
        basement of the Smithsonian Institution where he is director of the
        Marine Systems Laboratory, and another in a greenhouse of
        Washington's Old Soldiers Home. The problem was the proximity of
        the fresh and saltwater species. The result is a miniature Florida
        Everglades.
           ``The marsh may have been one of the most fragile of the systems
        because of the life that lives in the mud,'' explains Kathy Dyhr.
        ``But in the prototypes he found they had virtually no species
        extinctions except for the mosquitoes, to which everyone raised a
        cheer. But Walter said that's not so great because we need them for
        the fish. So they put up signs that said don't swat the
        mosquitoes.''
           In a world without chemicals, the biosphere settlers will rely
        on ladybugs, lacewings, marigolds and crop rotation to control
        pests. Human and animal waste and compost becomes fertilizer.
           ``The earth has its problems,'' says Carl Hodges of the
        University of Arizona who designed the human habitat. ``We've got
        acid rain problems, dirty air problems, and we've got carbon
        dioxide increasing at an alarming rate. Right now we don't have a
        research tool where we can control the global parameters, like CO2
        and the quality of the atmosphere. I see the big payoff of
        Biosphere II as learning how to do a better job of stewardship of
        Biosphere I.''
           The rain forest and its towering mountain and stream are the
        design of Dr. Ghillean Prance, director of the Kew Royal Botanical
        Gardens of London in association with the New York Botanical
        Garden. He patterned it after the dwindling rain forests of the
        Amazon.
           Dr. Peter Warshall of the University of Arizona patterned the
        savanna grasslands after those of South America, Australia and
        Africa. Plants and animals from those areas create the grasslands
        in Biosphere II.
           In all, some 3,800 species of plant and animal life are being
        invited to the party, and as in the original Eden, snakes. Says
        Dyhr, ``We may need a python in there, if we put in rodents which
        help aerate the soil and carry seeds.'' There will be no large
        predators, however. Not enough room. An ocelot needs two acres to
        range for food. There might be some mouse deer to help eat the
        grasses.
           Of almost as much interest as the individual biomes are the
        ecotones or tension areas where one biome yields reluctantly to the
        next. In these areas, where well-defined habitats change,
        scientists hope to learn how desert encroaches on more productive
        areas, what happens when fresh water turns brackish, what goes on
        when climates change.
           The idea of a biosphere, a living creation of nature with its
        own logic, its own rules, is not new. Probably the first to
        consider Earth and its environment as a biosphere was a Soviet
        scientist, V.I. Vernadsky, in 1926. The idea was further developed
        by James Lovelock and Lynn Margolis, who postulated, after Darwin,
        that the environment helps mediate species variation.
           In the beginning, or near it, they suggested that oxygen was
        sparse and so most of the early creatures were anaerobic bacteria
        that didn't need oxygen. But they gave off oxygen, and in the
        process poisoned their own environment so they had to retreat into
        the mud.
           Biosphere II may teach mankind enough so it does not suffer the
        same fate.
        """

    ro_big = """
        Fenomenul de încălzire globală nu este nicidecum ceva nou pentru planeta noastră. Perioadele de temperaturi foarte ridicate și cele glaciare alternează conform unui ciclu pe care oamenii de știință sunt încă departe de a-l înțelege, dar a cărui existentă e sprijinită de dovezi din ce în ce mai numeroase. Ca idee, în ultima jumătate de milion de ani planeta a trecut prin nu mai puțin de șapte perioade glaciare, iar acum 7.000 de ani am intrat într-o nouă perioadă de încălzire a atmosferei.
        Nu e ușor de acceptat ideea că societatea bazată pe utilizarea cărbunilor, petrolului și gazelor naturale e de vină pentru încălzirea globală, care anunță efecte dezastruoase din cauza schimbărilor climatice. Prin urmare, scepticii schimbărilor climatice contraatacă permanent, aducând în discuție o mulțime de alte posibile cauze care ar putea influența încălzirea globală.
        Unul dintre factorii naturali majori care pot influența radical temperatura Pământului este, evident, Soarele. Există posibilitatea ca o explozie solară să facă „ferfeniță” atmosfera planetei, iar radiațiile letale să treacă de bariera câmpului magnetic al planetei. Variația activității Soarelui poate duce la o creștere bruscă a temperaturii globale. Niciun om de știință nu neagă că aceste riscuri există. Însă aceste scenarii, care pot suna apocaliptic în filmele SF, nu se întâmplă brusc, ci în decursul unor decenii sau, mai degrabă, al unor secole. Iar, în ultimul secol, măsurătorile NASA și ale altor instituții arată că Soarele nostru a fost „cuminte”, iar influența asupra Pământului constantă. Doar pe la mijlocul anilor ’50 a avut o activitate ceva mai intensă, dar care, culmea, a coincis cu o scădere temporară a temperaturii medii globale.
    """

    summary = submodular_summarizer.summarize(en_doc, lang=Lang.EN)
    for sent in summary:
        print(sent)

    # dataset_parser = DUC2001Parser(Path.cwd() / "corpus" / "DUC2001")
    # summarize_duc2001_dataset(spacy_parser, dataset_parser, summarizer)

    # summarize_dataset(dataset_type=DatasetType.DUC_2002, summarizer=submodular_summarizer)


if __name__ == "__main__":

    main()
    # grid_search()
