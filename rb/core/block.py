from rb.parser.spacy_parser import SpacyParser
from rb.core.lang import Lang
from rb.core.text_element import TextElement
from rb.core.sentence import Sentence
from rb.core.text_element_type import TextElementType
from typing import List

from rb.utils.rblogger import Logger

logger = Logger.get_logger()


class Block(TextElement):


    def __init__(self, lang: Lang, text: str, index_in_container: int,
                 depth: int = TextElementType.BLOCK.value,
                 container: TextElement = None):

        TextElement.__init__(self, lang=lang, text=text, index_in_container=index_in_container,
                             depth=depth, container=container)
        sentences = SpacyParser.get_instance().parse_block(text, lang)
        for i, sentence in enumerate(sentences):
            self.components.append(Sentence(lang, sentence, i, container=self))
        
        self.has_coref = False
        # if sentences:
        #     try:
        #         doc = sentences[0].doc
        #         self.has_coref = doc._.has_coref
        #         if self.has_coref:
        #             words = {word.index_in_doc: word for sent in self.components for word in sent.components}
        #             self.coref_clusters = [CorefCluster(lang, cluster, words) for cluster in doc._.coref_clusters]
        #     except AttributeError:
        #         pass

    def get_sentences(self) -> List[Sentence]:
        return self.components
