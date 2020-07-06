import itertools
import numpy as np
import wmd

from pathlib import Path
from rb.core.lang import Lang
from rb.parser.spacy_parser import SpacyParser
from rb.summarization.enums.feature_type import FeatureType
from rb.summarization.utils.parser.custom_parser import CustomParser
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from typing import List, Union


nlp = None


def tokenize(parser: SpacyParser, doc_sentences: List[str], lang: Lang):
    """
    Tokenize a list of strings using spaCy pipeline
    :param parser:
    :param doc_sentences:
    :param lang:
    :return: tuples of index and tokens computed for every string in the list
    """

    lower_sentences = map(lambda t: t.lower(), doc_sentences)
    for i, tokens in enumerate(parser.get_tokens_lemmas(lower_sentences, lang)):
        tokens = [token.lemma_.lower() for token in tokens if not token.is_stop and token.is_alpha and len(token.text) > 2]
        if len(set(tokens)) > 3:
            yield (i, tokens)


def tokenize2(parser: SpacyParser, doc_sentences: List[str], lang: Lang):
    """
    Tokenize a list of strings using spaCy model
    :param parser:
    :param doc_sentences:
    :param lang:
    :return:
    """
    global nlp
    if not nlp:
        nlp = parser.get_model(lang)
        nlp.add_pipe(wmd.WMD.SpacySimilarityHook(nlp), last=True)

    lower_sentences = map(lambda t: t.lower(), doc_sentences)
    for i, sentence in enumerate(lower_sentences):
        # tokens = [token for token in nlp(sentence) if not token.is_stop and token.is_alpha and len(token.text) > 2]
        # if len(tokens) > 3:
        #     yield (i, tokens)
        if len(set(sentence.split())) > 3:
            yield (i, nlp(sentence))


def build_feature_matrix(docs: List, feature_type: Union[FeatureType, str], ngram_range=(1, 1), min_df=0.0, max_df=1.0):
    """
    :param docs: list of texts
    :param feature_type: 'binary', 'frequency', 'tfidf'
    :param ngram_range:
    :param min_df:
    :param max_df:
    :return:
    """

    if isinstance(feature_type, str):
        feature_type = FeatureType(feature_type.lower().strip())

    if feature_type == FeatureType.BINARY:
        vectorizer = CountVectorizer(binary=True, ngram_range=ngram_range, min_df=min_df, max_df=max_df)
    elif feature_type == FeatureType.FREQUENCY:
        vectorizer = CountVectorizer(binary=False, ngram_range=ngram_range, min_df=min_df, max_df=max_df)
    elif feature_type == FeatureType.TFIDF:
        vectorizer = TfidfVectorizer(ngram_range=ngram_range, min_df=min_df, max_df=max_df, use_idf=True)
    else:
        raise Exception("Wrong feature type entered. Possible values: 'binary', 'frequency', 'tfidf'")

    feature_matrix = vectorizer.fit_transform(docs).astype(float)

    return vectorizer, feature_matrix


def k_means(feature_matrix, no_clusters: int = 5, max_iter: int = 300):

    km = KMeans(n_clusters=no_clusters, max_iter=max_iter).fit(feature_matrix)

    return km, km.labels_


def get_indices(lst, value):
    """
    Returns all positional indices of a value in a list
    :param lst: list of values
    :param value: value to be searched for
    :return: list of positional indices
    """

    return [i for (el, i) in zip(lst, range(len(lst))) if value == el]


def list_intersection(lst1, lst2):
    """
    :param lst1: list of values
    :param lst2:  list of values
    :return: common unique values
    """

    return list(set(lst1) & set(lst2))


def write_list_to_file(file_path: Path, var_list: List[str]):
    with open(str(file_path), 'w', encoding='utf-8') as f:
        for el in var_list:
            f.write(el + '\n')


def write_str_to_file(file_path: Path, var_str: str):
    with open(str(file_path), 'w', encoding='utf-8') as f:
        f.write(var_str)


def append_list_to_file(file_path: Path, var_list: List[str]):
    with open(str(file_path), 'a', encoding='utf-8') as f:
        for el in var_list:
            f.write(el + '\n')


"""
            Similarity metrics below 
"""

DELTA = 0.85
ETA = 0.4


def most_similar_token(joint_token, tokens):
    max_sim = -1.0
    sim_token = None

    if not joint_token or not joint_token.vector_norm:
        return sim_token, max_sim

    for ref_token in tokens:
        # check if token is a valid spaCy object and has a word vector
        if ref_token and ref_token.vector_norm:
            sim = joint_token.similarity(ref_token)
            if sim > max_sim:
                max_sim = sim
                sim_token = ref_token
    return sim_token, max_sim


def word_order_vector(tokens, joint_tokens):
    wovec = np.zeros(len(joint_tokens))
    for i, joint_token in enumerate(joint_tokens):
        if joint_token in tokens:
            wovec[i] = tokens.index(joint_token)
        else:
            sim_token, max_sim = most_similar_token(joint_token, tokens)
            if max_sim > ETA:
                wovec[i] = tokens.index(sim_token)
            else:
                wovec[i] = 0
    return wovec


def word_order_similarity(doc1, doc2):
    """

    :param doc1: list of spaCy tokens
    :param doc2: list of spaCy tokens
    :return: word order similarity score
    """

    doc1_dict = {token.text: token for token in doc1}
    doc2_dict = {token.text: token for token in doc2}
    joint_tokens = {**doc1_dict, **doc2_dict}.values()  # dictionary merge syntax valid in python >= 3.5
    r1 = word_order_vector(doc1, joint_tokens)
    r2 = word_order_vector(doc2, joint_tokens)

    # check if both arrays have only zeros
    if np.count_nonzero(r1) == 0 and np.count_nonzero(r2) == 0:
        return 0.0

    return 1.0 - (np.linalg.norm(r1 - r2) / np.linalg.norm(r1 + r2))


def word_embeddings_similarity(doc1, doc2):
    vector1 = np.zeros(300)
    vector2 = np.zeros(300)
    for token in doc1:
        # if not token.is_stop and token.is_alpha:
        vector1 = vector1 + token.vector
    vector1 = np.divide(vector1, len(doc1))
    for token in doc2:
        # if not token.is_stop and token.is_alpha:
        vector2 = vector2 + token.vector
    vector2 = np.divide(vector2, len(doc2))

    if np.count_nonzero(vector1) == 0 or np.count_nonzero(vector2) == 0:
        return 0.0

    return np.dot(vector1, vector2) / (np.linalg.norm(vector1 * np.linalg.norm(vector2)))


def combined_similarity(docs):
    """
    Linear combination similarity between word embeddings similarity and word order similarity.
    :param docs:
    :return:
    """

    n = len(docs)
    similarity_matrix = np.identity(n, float)
    for i, j in itertools.combinations(range(n), 2):
        sim = DELTA * word_embeddings_similarity(docs[i], docs[j]) + \
              (1.0 - DELTA) * word_order_similarity(docs[i], docs[j])
        similarity_matrix[i, j] = similarity_matrix[j, i] = sim
    return similarity_matrix


def sentence_similarity(sentence_1: str, sentence_2: str, parser: SpacyParser, lang: Lang = Lang.EN):
    nlp = parser.get_model(lang)
    doc1 = nlp(sentence_1)
    doc2 = nlp(sentence_2)
    return DELTA * word_embeddings_similarity(doc1, doc2) + (1.0 - DELTA) * word_order_similarity(doc1, doc2)


def generate_gensim_system_summaries(spacy_parser, dataset_parser):

    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings(action="ignore", category=UserWarning, module="gensim")
        warnings.filterwarnings(action="ignore", category=DeprecationWarning)
        from gensim.summarization import summarize

    i = 0
    for _, docs in dataset_parser.get_docs_with_summary().items():
        for doc in docs:
            print("Processing %s in iteration %d" % (doc, i))
            doc_text = dataset_parser.get_document_text(doc)
            model_summary_text = dataset_parser.get_model_summary(doc)

            model_summary_as_list = spacy_parser.tokenize_sentences(model_summary_text)
            model_summary_as_str = '\n'.join(model_summary_as_list)

            gensim_summary_as_str = '\n'.join(summarize(doc_text, ratio=0.2, split=True))

            model_summary_filename = "duc2001.A." + "{0:0=3d}".format(i + 1) + ".txt"
            system_summary_filename = "duc2001." + "{0:0=3d}".format(i + 1) + ".txt"

            rouge_path = Path.cwd() / ".." / "rouge"
            write_str_to_file(rouge_path / "model_summaries" / model_summary_filename, model_summary_as_str)
            write_str_to_file(rouge_path / "system_summaries" / system_summary_filename, gensim_summary_as_str)

            i = i + 1


def main():
    parser = CustomParser().get_instance()
    sentence1 = "A quick brown dog jumps over the lazy fox."
    sentence2 = "A quick brown fox jumps over the lazy dog."

    # tokenized_sentences = list(tokenize2(parser, [sentence1, sentence2], Lang.EN))

    # print(word_order_similarity(tokenized_sentences[0][1], tokenized_sentences[1][1]))

    # sentence1 = "Obama speaks to the media in Illinois."
    # sentence2 = "The president greets the press in Chicago."
    # res = sentence_similarity(sentence1, sentence2, parser, Lang.EN)
    # print(res)

    # dataset_parser = DUC2001Parser(Path.cwd() / ".." / "corpus"/ "DUC2001")
    # generate_gensim_system_summaries(parser, dataset_parser)


if __name__ == "__main__":

    main()
