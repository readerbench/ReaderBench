from rb.core.span import Span
from rb.core.word import Word
from rb.core.lang import Lang
from rb.core.text_element import TextElement
from neuralcoref.neuralcoref import Cluster
from typing import List, Dict


class CorefCluster():

    def __init__(self, lang: Lang, nc_cluster: Cluster, words: Dict[int, Word]):
        self.lang = lang
        self.index = nc_cluster.i
        self.main = Span.from_spacy_span(lang, nc_cluster.main, words)
        self.mentions = [Span.from_spacy_span(lang, mention, words) for mention in nc_cluster.mentions]
        # update the words
        for mention in nc_cluster.mentions:
            m_words = [words[i] for i in range(mention.start, mention.end)]
            for word in m_words:
                word.in_coref = True
                word.coref_clusters.append(self)
