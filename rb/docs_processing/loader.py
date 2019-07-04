from os import listdir
from os.path import isfile, join

import xml.etree.ElementTree as ET

from rb.docs_processing.article import Article
from rb.docs_processing.graph import Graph
from rb.core.lang import Lang



def parse_xml_file(file_path: str, graph: Graph) -> Article:
    tree = ET.parse(file_path)
    root = tree.getroot()

    abstract = root[0][0].text
    title = root[1][0].text
    date = root[1][1].text
    source = root[1][2].text
    authors = []
    for element in root[1][4]:
        authors.append(element.text)
    article = Article(title, abstract, source, authors, date, Lang.EN, graph)
    return article


def load_directory_xmls(directory_path: str) -> Graph:
    graph = Graph()
    for f in listdir(directory_path):
        if isfile(join(directory_path, f)):
            print(join(directory_path, f))
            parse_xml_file(join(directory_path, f), graph)
    return graph

# load_directory_xmls("C:\\Users\\Dragos\\Documents\\Facultate-Munca\\onlinedatasetexplorer\\AI_grub")

