from rb.similarity.word2vec import Word2Vec
from rb.similarity.lda import LDA
from rb.similarity.lsa import LSA
from rb.similarity.vector_model import VectorModel
from rb.core.lang import Lang
from rb.comprehension.comprehension_model import ComprehensionModel
from rb.comprehension.utils.graph.cm_node_type import CmNodeType
from rb.comprehension.utils.graph.cm_node_do import CmNodeDO
from rb.core.word import Word
from rb.comprehension.classifiers.amoc_classifier import random_forest_classification

import time

from copy import deepcopy

from os import listdir
from os.path import isfile, join
import xml.etree.ElementTree as ET


ACTIVATION_SCORE = 0.3
MAX_ACTIVE_CONCEPTS = 20
MAX_DICTIONARY_EXPANSION = 5


def comprehension_example(test_string, w2v, lsa, lda):
    t1 = time.time()
    global ACTIVATION_SCORE
    global MAX_ACTIVE_CONCEPTS
    global MAX_DICTIONARY_EXPANSION
    # cm = ComprehensionModel(test_string, Lang.EN, [w2v, lsa, lda],
    cm = ComprehensionModel(test_string, Lang.EN, [w2v],
                            ACTIVATION_SCORE, MAX_ACTIVE_CONCEPTS, MAX_DICTIONARY_EXPANSION)

    for index in range(cm.get_total_number_of_phrases()):
        now = time.time()
        sentence = cm.get_sentence_at_index(index)

        # syntactic_indexer = cm.get_syntactic_indexer_at_index(index)
        # current_syntactic_graph = syntactic_indexer.get_cm_graph(CmNodeType.TextBased)
        current_syntactic_graph = cm.sentence_graphs[index]
        current_graph = cm.current_graph

        current_graph.combine_with_syntactic_links(current_syntactic_graph, sentence, cm.semantic_models,
                                                   cm.max_dictionary_expansion)

        cm.current_graph = current_graph
        middle = time.time()
        cm.apply_page_rank(index)
        then = time.time()
        print("Page rank a durat {}".format(int(then - middle)))
        print("A durat {}".format(int(then - now)))

    # print(cm.current_graph)
    t2 = time.time()
    print("In total a durat {}".format(int(t2 - t1)))
    return cm.history_keeper.compute_statistics()


def generate_closest_words(model: VectorModel, lang: Lang):
    similar_words = {}
    counter = 0
    for key in model.vectors.keys():
        similar_words[key] = model.most_similar(key, 20, 0.0)
        counter += 1
        if counter % 100 == 0:
            print(model.name + " done " + str(counter) + "/" + str(len(model.vectors)))

    import csv
    with open(model.name + "_" + lang.value + "_" + "most_similar.csv", 'w') as f:
        for key, value in similar_words.items():
            f.write("%s,%s\n" % (key, ",".join(map(str, value))))


def get_text_from_xml_file(filepath):
    tree = ET.parse(filepath)
    root = tree.getroot()
    body = root[1]
    return " ".join([p.text for p in body])


def generate_low_high_dataset(input_path, output_file, extension):
    only_extension_files = [join(input_path, f) for f in listdir(input_path)
                            if isfile(join(input_path, f)) and f.endswith(extension)]
    w2v = Word2Vec('coca', Lang.EN)
    lsa = LSA('coca', Lang.EN)
    lda = LDA('coca', Lang.EN)
    with open(output_file, "a", encoding="utf-8") as of:
        if extension == ".txt":
            write_txt_result(lda, lsa, of, only_extension_files, w2v)
        elif extension == ".xml":
            write_xml_result(lda, lsa, of, only_extension_files, w2v)


def write_txt_result(lda, lsa, of, only_extension_files, w2v):
    for index, file_name in enumerate(only_extension_files):
        with open(file_name, "r", encoding="utf-8") as f:
            content = f.read()
            results = comprehension_example(content, w2v, lsa, lda)
            of.write(str(index % 2) + ", " + file_name + ", " + ", ".join([str(x) for x in results]) + "\n")


def write_xml_result(lda, lsa, of, only_extension_files, w2v):
    for index, file_name in enumerate(only_extension_files):
        content = get_text_from_xml_file(file_name)
        results = comprehension_example(content, w2v, lsa, lda)
        of.write(str(index % 2) + ", " + file_name + ", " + ", ".join([str(x) for x in results]) + "\n")


def call_generate():
    with open("amoc_from_xml_dataset.csv", "w") as f:
        f.write("class,name,mean_activation_for_active_nodes,mean_activation_for_all_nodes,mean_degree_centrality,"
                "mean_closeness_centrality,mean_betweenness_centrality,mean_harmonic_centrality,"
                "mean_active_nodes_percentage,mean_density,mean_modularity")
        f.write("\n")
    generate_low_high_dataset("C:\\Users\\Dragos\\Downloads\\texts", "amoc_from_xml_dataset.csv", ".xml")


def run_classifiers():
    random_forest_classification("amoc_results/amoc_lsa_lda_w2v_dataset.csv")


def main():
    w2v = Word2Vec('coca', Lang.EN)
    lsa = LSA('coca', Lang.EN)
    lda = LDA('coca', Lang.EN)
    # generate_closest_words(w2v, Lang.EN)
    # generate_closest_words(lsa, Lang.EN)
    # generate_closest_words(lda, Lang.EN)
    # test_comprehension("A young knight rode through the forest. The knight was unfamiliar with the country. Suddenly, a dragon appeared. The dragon was kidnapping a beautiful princess. The knight wanted to free the princess. The knight wanted to marry the princess. The knight hurried after the dragon. They fought for life and death. Soon, the knight's armor was completely scorched. At last, the knight killed the dragon. The knight freed the princess. The princess was very thankful to the knight. She married the knight.")
    comprehension_example("""George III, the king of England, said that there had to be a tax on something to prove that the British had the right to tax. So there was still a small tax on tea. The colonists remained firm. They would not pay any tax passed by Parliament. Colonial women refused to buy or serve tea.
      British merchants were not selling much tea. So Parliament passed a law that greatly lowered its price. Boatloads of tea were sent to America. Since it was cheaper than ever, the British thought that surely the colonists would buy tea now!
      They were wrong. Tea was burned. Tea was left to rot. Ships loaded with it were not allowed in ports. In Boston the Sons of Liberty dressed up as Indians. Late at night they went to Boston Harbor and threw more than 300 chests of tea into the water. This action was called the Boston Tea Party.""",
                          w2v, lsa, lda)
    # w = Word.from_str(Lang.EN, "yes")
    # n = CmNodeDO(w, CmNodeType.TextBased)
    # ss = deepcopy({n: 1.0})


def run_grid_search():
    global ACTIVATION_SCORE
    global MAX_DICTIONARY_EXPANSION
    global MAX_ACTIVE_CONCEPTS
    activation_score_list = [0.3, 0.35, 0.4, 0.45, 0.5]
    max_active_concepts_list = list(range(7, 31))
    max_dictionary_expansion_list = list(range(1, 8))
    max_acc = 0.0
    best_activation = 0
    best_active_concepts = 0
    best_max_dictionary_exp = 0
    for activation_score in activation_score_list:
        ACTIVATION_SCORE = activation_score
        for mac in max_active_concepts_list:
            MAX_ACTIVE_CONCEPTS = mac
            for mde in max_dictionary_expansion_list:
                MAX_DICTIONARY_EXPANSION = mde
                call_generate()
                acc = 0.0
                for _ in range(10):
                    acc += random_forest_classification("amoc_from_xml_dataset.csv")
                if max_acc < acc / 10:
                    max_acc = acc / 10
                    best_activation = activation_score
                    best_active_concepts = mac
                    best_max_dictionary_exp = mde
                with open("amoc_runs.txt", "a", encoding="UTF-8") as f:
                    f.write(str(acc / 10) + " " + str(activation_score) + " " + str(mac ) + " " + str(mde) + "\n")
    with open("amoc_runs.txt", "a", encoding="UTF-8") as f:
        f.write(str(max_acc) + " " + str(best_activation) + " " + str(best_active_concepts) + " "
                + str(best_max_dictionary_exp) + "\n")
    print(max_acc, best_activation, best_active_concepts, best_max_dictionary_exp)


if __name__ == "__main__":
    # main()
    # call_generate()
    run_grid_search()
    # run_classifiers()
