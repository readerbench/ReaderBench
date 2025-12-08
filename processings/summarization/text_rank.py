from rb.core.text_element import TextElement
from rb.cna.cna_graph import CnaGraph
from rb.core.lang import Lang
from rb.core.sentence import Sentence
from typing import Union, List, Tuple
from rb.similarity.vector_model_factory import get_default_model
from rb.core.document import Document
from operator import itemgetter

def get_sentence_scores(doc: Document) -> List[Tuple[int, Sentence, float]]:
    model = get_default_model(doc.lang)
    graph = CnaGraph(doc, [model])
    sentences = sorted([
        (i, sentence, graph.importance[sentence]) 
        for i, sentence in enumerate(doc.get_sentences())], 
        key=itemgetter(2),
        reverse=True)
    return sentences

def summarize(doc: Union[str, TextElement], lang: Lang = None, no_sentences=3) -> str:
    if isinstance(doc, str):
        assert lang is not None, "lang parameter is required for str docs"
        doc = Document(lang, doc)
    else:
        lang = doc.lang
    sentences = get_sentence_scores(doc)
    if len(sentences) > no_sentences:
        sentences = sentences[:no_sentences]
    return " ".join([sent.text for i, sent, score in sentences])
