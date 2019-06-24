from rb.similarity.word2vec import Word2Vec
from rb.similarity.lda import LDA
from rb.similarity.lsa import LSA
from rb.similarity.vector_model import VectorModel
from rb.core.lang import Lang
from rb.comprehension.comprehension_model import ComprehensionModel
from rb.comprehension.utils.graph.cm_node_type import CmNodeType
from rb.comprehension.utils.graph.cm_node_do import CmNodeDO
from rb.core.word import Word

import time

from copy import deepcopy

def test_comprehension(test_string):
    w2v = Word2Vec('coca', Lang.EN)
    lsa = LSA('coca', Lang.EN)
    lda = LDA('coca', Lang.EN)

    t1 = time.time()
    cm = ComprehensionModel(test_string, Lang.EN, [w2v, lsa, lda],
                        0.3, 20, 5)
    
    for index in range(cm.get_total_number_of_phrases()):
        now = time.time()
        sentence = cm.get_sentence_at_index(index)

        # syntactic_indexer = cm.get_syntactic_indexer_at_index(index)
        # current_syntactic_graph = syntactic_indexer.get_cm_graph(CmNodeType.TextBased)
        current_syntactic_graph = cm.sentence_graphs[index]
        current_graph = cm.current_graph

        current_graph.combine_with_syntactic_links(current_syntactic_graph, sentence, cm.semantic_models, cm.max_dictionary_expansion)

        cm.current_graph = current_graph
        middle = time.time()
        cm.apply_page_rank(index)
        then = time.time()
        print("Page rank a durat {}".format(int(then - middle)))
        print("A durat {}".format(int(then - now)))

    print(cm.current_graph)
    t2 = time.time()
    print("In total a durat {}".format(int(t2 - t1)))


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
            f.write("%s,%s\n"%(key, ",".join(map(str, value))))


w2v = Word2Vec('coca', Lang.EN)
lsa = LSA('coca', Lang.EN)
lda = LDA('coca', Lang.EN)

# generate_closest_words(w2v, Lang.EN)
# generate_closest_words(lsa, Lang.EN)
# generate_closest_words(lda, Lang.EN)
test_comprehension("A young knight rode through the forest. The knight was unfamiliar with the country. Suddenly, a dragon appeared. The dragon was kidnapping a beautiful princess. The knight wanted to free the princess. The knight wanted to marry the princess. The knight hurried after the dragon. They fought for life and death. Soon, the knight's armor was completely scorched. At last, the knight killed the dragon. The knight freed the princess. The princess was very thankful to the knight. She married the knight.")

#w = Word.from_str(Lang.EN, "yes")
#n = CmNodeDO(w, CmNodeType.TextBased)
#ss = deepcopy({n: 1.0})