from rb.similarity.word2vec import Word2Vec
from rb.similarity.lda import LDA
from rb.similarity.lsa import LSA
from rb.core.lang import Lang
from rb.core.document import Document

from rb.docs_processing.loader import load_directory_xmls, load_directory_nic_xls, load_directory_santiago_xls
from rb.docs_processing.coauthorship import CoAuthorship
from rb.docs_processing.metrics import GraphMetrics

from rb.processings.keywords.keywords_extractor import KeywordExtractor


def test_similarities():
    doc1 = Document(Lang.EN, """Background. Being able to maintain personal hygiene plays a crucial role for independent living in old age or when suffering from disabilities. Within the European project 
ICT Supported Bath Robots (I-SUPPORT) an intelligent robotic shower system is being developed, which enables patients to shower independently at home or in institutionalized settings. 
Objective. The aim of this contribution is
the identification of ethical issues in the development of a robotic shower system utilizing the model for the ethical evaluation of socio-technical arrangements (MEESTAR). Material and methods. In I-SUPPORT a variety of concepts and methods are implemented in order to achieve technology acceptance such as user-centered requirements analysis, usability-tests and analysis of sociocultural and ethical aspects. This article reports 
the analysis of focus groups with 14 older 
adults and 9 professional caregivers utilizing MEESTAR as a heuristic approach for analyzing sociotechnical arrangements and identifying ethical problems. 
Results and discussion. The MEESTAR procedure was adapted to the research question and client groups and implemented as a discursive method. This gave an insight into the meaning and background of ethical aspects and also a deeper insight into nursing processes as well as the requirements which the system should fulfil. Shortcomings are that the ethical dimensions are not everyday language and the time restrictions. In the next step a standardized assessment instrument will be developed and piloted. 
""")
    doc2 = Document(Lang.EN, """Background. Being able to maintain personal hygiene plays a crucial role for independent living in old age or when suffering from disabilities. Within the European project
ICT Supported Bath Robots (I-SUPPORT) an intelligent robotic shower system is being developed, which enables patients to shower independently at home or in institutionalized settings.
Objective. The aim of this contribution is
the identification of ethical issues in the development of a robotic shower system utilizing the model for the ethical evaluation of socio-technical arrangements (MEESTAR). Material and methods. In I-SUPPORT a variety of concepts and methods are implemented in order to achieve technology acceptance such as user-centered requirements analysis, usability-tests and analysis of sociocultural and ethical aspects. This article reports
the analysis of focus groups with 14 older
adults and 9 professional caregivers utilizing MEESTAR as a heuristic approach for analyzing sociotechnical arrangements and identifying ethical problems.
Results and discussion. The MEESTAR procedure was adapted to the research question and client groups and implemented as a discursive method. This gave an insight into the meaning and background of ethical aspects and also a deeper insight into nursing processes as well as the requirements which the system should fulfil. Shortcomings are that the ethical dimensions are not everyday language and the time restrictions. In the next step a standardized assessment instrument will be developed and piloted.""")

    w2v = Word2Vec('coca', Lang.EN)
    lsa = LSA('coca', Lang.EN)
    lda = LDA('coca', Lang.EN)
    print(w2v.similarity(doc1, doc2))
    print(lsa.similarity(doc1, doc2))
    print(lda.similarity(doc1, doc2))


def write_to_file_cluster_results(filename, results):
    with open(filename, 'w', encoding="UTF-8") as f:
        kw_extractor = KeywordExtractor()
        for result in results:
            f.write("! Articles in the cluster:\n" +
                    result[0] +
                    "\n--------\n! All the abstracts in the cluster:\n"
                    + result[1] + "\n~~~~~~~~\n! Keywords:\n" +
                    str(kw_extractor.extract_keywords(result[1], Lang.EN)) +
                    "\n********\n")


if __name__ == "__main__":
    # test_similarities()
    w2v = Word2Vec('coca', Lang.EN)
    lsa = LSA('coca', Lang.EN)
    lda = LDA('coca', Lang.EN)

    # graph = load_directory_xmls("C:\\Users\\Dragos\\Documents\\Facultate-Munca\\onlinedatasetexplorer\\AI_grub")
    # graph = load_directory_xmls("C:\\Users\\Dragos\\Downloads\\lak_2011_2019.tar\\lak_2011_2019")
    graph = load_directory_nic_xls("C:\\Users\\Dragos\\Documents\\Facultate-Munca\\_Tables\\combined")
    # graph = load_directory_santiago_xls("C:\\Users\\Dragos\\Documents\\Facultate-Munca\\_Santiago")
    graph.semantic_models = [w2v, lsa, lda]
    print("Loading done")
    graph.extract_authors_dict()
    print("Author extraction done")
    graph.build_edges_and_construct_author_rankings_graph(include_authors=False)
    print("Build edges done")

    keyword_extractor = KeywordExtractor()
    all_abstracts = " ".join([article.abstract for article in graph.articles_set])
    with open("all_abstracts.txt", "w", encoding="UTF-8") as f:
        f.write(all_abstracts)
    print(keyword_extractor.extract_keywords(all_abstracts, Lang.EN))
    # graph_metrics = GraphMetrics(graph)
    # results_number = graph_metrics.get_top_clusters_of_articles_by_number_of_elements(10)
    # results_degree = graph_metrics.get_top_clusters_of_articles_by_degree_of_elements(10)
    # write_to_file_cluster_results("cluster_number.txt", results_number)
    # write_to_file_cluster_results("cluster_degree.txt", results_degree)
    # graph_metrics.perform_articles_agglomerative_clustering()
    # graph_metrics.perform_authors_agglomerative_clustering()
    # print(len(graph.authors_set))
    # print(graph_metrics.get_top_n_articles_by_closeness(10))


    # print(len(graph.articles_set))
    # co_authorship_client = CoAuthorship(graph)
    # co_authorship_client.print_results()
    # for x, y in graph.adjacent_list.items():
    #     print(x, y)
    #     break

    # print(graph.authors_ranking_graph[graph.authors_dict["Nicolae Nistor"]])

    # graph.plot_histogram_of_articles_distance()
    # print("Histogram done")
    # with open('santiago_authors.txt', 'w', encoding='utf8') as f:
    #     authors_list = graph.get_authors_by_type_degree(100)
    #     for index, info in enumerate(authors_list):
    #         author = info[0]
    #         degree = info[1]
    #         f.write("{}. Author name: {}, with degree {}. Number of articles in this corpus {}.\n"
    #                 .format(index + 1, author.name, degree, len(author.articles)))
    #
    # with open('santiago_articles.txt', 'w', encoding='utf8') as f:
    #     articles_list = graph.get_articles_by_type_degree(100)
    #     for index, info in enumerate(articles_list):
    #         f.write("{}. Article name: {}, with degree {} and number of edges {}\n".format(index + 1, info[0].title,
    #                                                                                        info[1], info[2]))
    #
    # with open('santiago_articles_closeness.txt', 'w', encoding='utf8') as f:
    #     articles_closeness_rankings = graph_metrics.get_top_n_articles_by_closeness(100);
    #     for index, info in enumerate(articles_closeness_rankings):
    #         f.write("{}. Article name: {}, closeness: {}\n".format(index + 1, info[0].title, info[1]))
    #
    # with open('santiago_articles_betweenness.txt', 'w', encoding='utf8') as f:
    #     articles_betweenness_rankings = graph_metrics.get_top_n_articles_by_betweenness(100);
    #     for index, info in enumerate(articles_betweenness_rankings):
    #         f.write("{}. Article name: {}, betweenness: {}\n".format(index + 1, info[0].title, info[1]))
    #
    # with open('santiago_authors_closeness.txt', 'w', encoding='utf8') as f:
    #     authors_closeness_rankings = graph_metrics.get_top_n_authors_by_closeness(100);
    #     for index, info in enumerate(authors_closeness_rankings):
    #         f.write("{}. Author name: {}, closeness: {}\n".format(index + 1, info[0].name, info[1]))
    #
    # with open('santiago_authors_betweenness.txt', 'w', encoding='utf8') as f:
    #     authors_betweenness_rankings = graph_metrics.get_top_n_authors_by_betweenness(100);
    #     for index, info in enumerate(authors_betweenness_rankings):
    #         f.write("{}. Author name: {}, betweenness: {}\n".format(index + 1, info[0].name, info[1]))

    # with open('articles_distances.csv', 'w', encoding='utf8') as f:
    #     articles_pairs = graph.get_distances_between_articles()
    #     for pair in articles_pairs:
    #         f.write("{}$ {}$ {}\n".format(pair[0].title, pair[1].title, pair[2]))
