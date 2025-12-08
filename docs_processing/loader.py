from os import listdir
from os.path import isfile, join

import xml.etree.ElementTree as ET
import xlrd

from rb.docs_processing.article import Article, create_article_and_add_it_to_its_authors_and_graph
from rb.docs_processing.graph import Graph
from rb.core.lang import Lang


def parse_xml_file_and_add_article_to_graph(file_path: str, graph: Graph) -> Article:
    tree = ET.parse(file_path)
    root = tree.getroot()

    abstract = root[0][0].text
    title = root[1][0].text.replace('\n', ' ')
    date = root[1][1].text
    source = root[1][2].text
    authors = []
    for element in root[1][4]:
        authors.append(element.text)
    create_article_and_add_it_to_its_authors_and_graph(title, abstract, source, authors, date, Lang.EN, graph)


def parse_santiago_xls_file_and_add_articles_to_graph(file_path: str, graph: Graph):
    workbook = xlrd.open_workbook(file_path)
    sheet = workbook.sheet_by_index(0)

    for i in range(1, 217):
        date = "01-01-" + str(int(sheet.cell_value(i, 1)))
        title = sheet.cell_value(i, 4)
        authors = sheet.cell_value(i, 2).split(";")
        abstract = sheet.cell_value(i, 12)
        source = "SANTIAGO"
        if not abstract or not title or not authors:
            continue
        create_article_and_add_it_to_its_authors_and_graph(title, abstract, source, authors, date, Lang.EN, graph)


def load_directory_xmls(directory_path: str) -> Graph:
    graph = Graph()
    for f in listdir(directory_path):
        file_path = join(directory_path, f)
        if isfile(file_path):
            print(file_path)
            parse_xml_file_and_add_article_to_graph(file_path, graph)
    return graph


def parse_xls_file_and_add_articles_to_graph(file_path: str, graph: Graph):
    workbook = xlrd.open_workbook(file_path)
    sheet = workbook.sheet_by_index(0)

    for i in range(1, sheet.nrows):
        date = "01-01-" + str(int(sheet.cell_value(i, 1)))
        title = sheet.cell_value(i, 2).replace('\n', ' ')
        authors = sheet.cell_value(i, 3).split(",")
        abstract = sheet.cell_value(i, 4)
        source = "NIC"
        if not abstract or not title or not authors:
            continue
        create_article_and_add_it_to_its_authors_and_graph(title, abstract, source, authors, date, Lang.EN, graph)


def load_directory_nic_xls(directory_path: str) -> Graph:
    graph = Graph([])
    for f in listdir(directory_path):
        file_path = join(directory_path, f)
        if isfile(file_path):
            print(file_path)
            parse_xls_file_and_add_articles_to_graph(file_path, graph)
    return graph


def load_directory_santiago_xls(directory_path: str) -> Graph:
    graph = Graph([])
    for f in listdir(directory_path):
        file_path = join(directory_path, f)
        if isfile(file_path):
            print(file_path)
            parse_santiago_xls_file_and_add_articles_to_graph(file_path, graph)
    return graph

# load_directory_xmls("C:\\Users\\Dragos\\Documents\\Facultate-Munca\\onlinedatasetexplorer\\AI_grub")

