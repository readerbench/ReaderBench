import matplotlib.pyplot as plt
import networkx as nx

from rb.summarization.utils.utils import *
from rb.summarization.systems.summarizer_abc import *
from sklearn.metrics.pairwise import cosine_similarity
from typing import List


use_word_embeddings = False


def build_graph(tokenized_sentences: List):
    graph = nx.Graph()
    nodes = range(len(tokenized_sentences))
    graph.add_nodes_from(nodes)

    global use_word_embeddings
    if use_word_embeddings:
        similarity_matrix = combined_similarity(tokenized_sentences)
    else:
        sentences = [' '.join(tokens) for tokens in tokenized_sentences]
        vec, feature_matrix = build_feature_matrix(docs=sentences,
                                                   feature_type=FeatureType.TFIDF,
                                                   ngram_range=(1, 2),
                                                   min_df=0.1,
                                                   max_df=0.9)
        similarity_matrix = cosine_similarity(feature_matrix)

    node_pairs = list(itertools.combinations(nodes, 2))
    for u, v in node_pairs:
        weight = similarity_matrix[u, v]
        graph.add_edge(u, v, weight=weight)

    return graph


class TextRank(Summarizer):

    def __init__(self):
        Summarizer.__init__(self)

    def summarize(self, doc, lang=Lang.EN, parser=None, ratio=0.2, word_count=None) -> Iterable[str]:
        if not parser:
            parser = self.parser

        # Split the document into an iterable collection of strings.
        doc_sentences = parser.tokenize_sentences(doc)

        # Preprocess the sentences by cleaning tags and whitespaces.
        doc_sentences = list(map(lambda sentence: parser.preprocess(sentence, lang.value), doc_sentences))

        global use_word_embeddings
        if use_word_embeddings:
            doc_indices, tokenized_sentences = zip(*tokenize2(parser, doc_sentences, lang))
        else:
            doc_indices, tokenized_sentences = zip(*tokenize(parser, doc_sentences, lang))

        summary_size = int(len(doc_sentences) * ratio) if word_count is None else 1
        tokenized_sentences_len = len(tokenized_sentences)
        assert 0 < summary_size < tokenized_sentences_len

        get_doc_index = dict(zip(range(len(doc_indices)), doc_indices))
        # docs_str = [' '.join(map(lambda token: token.lemma_.lower(), sent)) for sent in tokenized_sentences]
        # vec, feature_matrix = build_feature_matrix(docs_str, FeatureType.TFIDF, ngram_range=(1, 2))
        similarity_graph = build_graph(tokenized_sentences)

        # similarity_matrix = (feature_matrix * feature_matrix.T)
        # similarity_graph = networkx.from_scipy_sparse_matrix(similarity_matrix)

        # similarity_matrix = cosine_similarity(feature_matrix)
        # similarity_graph = nx.from_numpy_array(similarity_matrix)

        # Plot graph
        # fig, ax = plt.subplots()
        # nx.draw_networkx(similarity_graph, ax=ax)
        # plt.show()

        scores = nx.pagerank(similarity_graph, weight="weight")

        ranked_sentences = sorted(scores, key=scores.get, reverse=True)

        top_sentences_indices = ranked_sentences[:summary_size]
        top_sentences_indices.sort()

        # for i in top_sentences_indices:
        #     yield doc_sentences[get_doc_index[i]]

        summary_sentences = map(lambda i: doc_sentences[get_doc_index[i]], top_sentences_indices)
        # return {'summary': ' '.join(summary_sentences)}
        return list(summary_sentences)


def main():

    textrank_summarizer = TextRank()

    doc = """
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

    doc2 = """
            Elephants are large mammals of the family Elephantidae
            and the order Proboscidea. Two species are traditionally recognised,
            the African elephant and the Asian elephant. Male
            African elephants are the largest extant terrestrial animals. All
            elephants have a long trunk used for many purposes,
            particularly breathing, lifting water and grasping objects..  
        """

    # output = textrank_summarizer.summarize(doc)
    # print(output)
    # for sent in summary:
    #     print(sent)

    # summarize_dataset(dataset_type=DatasetType.DUC_2002, summarizer=textrank_summarizer)


if __name__ == "__main__":

    main()
