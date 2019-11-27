from rb.similarity.word2vec import Word2Vec
from rb.similarity.lda import LDA
from rb.similarity.lsa import LSA
from rb.core.lang import Lang

from rb.docs_processing.loader import load_directory_xmls


if __name__ == "__main__":
    w2v = Word2Vec('coca', Lang.EN)
    lsa = LSA('coca', Lang.EN)
    lda = LDA('coca', Lang.EN)

    graph = load_directory_xmls("C:\\Users\\Dragos\\Documents\\Facultate-Munca\\onlinedatasetexplorer\\AI_grub")
    # graph = load_directory_xmls("C:\\Users\\Dragos\\Downloads\\lak_2011_2019.tar\\lak_2011_2019")    
    graph.extract_authors_dict()
    graph.build_edges([w2v, lsa, lda])

    # for x, y in graph.adjacent_list.items():
    #     print(x, y)
    #     break

    with open('authors2.txt', 'w', encoding='utf8') as f:
        authors_list = graph.get_authors_by_type_degree(100)
        for index, info in enumerate(authors_list):
            f.write("{}. Author name: {}, with type degree {}\n".format(index + 1, info[0].name, info[1]))

    
    with open('articles.txt', 'w', encoding='utf8') as f:
        articles_list = graph.get_articles_by_type_degree(100)
        for index, info in enumerate(articles_list):
            f.write("{}. Article name: {}, with type degree {}\n".format(index + 1, info[0].title, info[1]))

    # with open('articles_distances.csv', 'w', encoding='utf8') as f:
    #     articles_pairs = graph.get_distances_between_articles()
    #     for pair in articles_pairs:
    #         f.write("{}$ {}$ {}\n".format(pair[0].title, pair[1].title, pair[2]))