from rb.core.lang import Lang
from rb.core.sentence import Sentence
from rb.core.word import Word
from rb.core.document import Document

from rb.comprehension.utils.distance_strategies.cm_syntactic_graph import CmSyntacticGraph

from copy import deepcopy

from typing import Tuple, List
Dependency = Tuple[Word, Word, str]
Dependencies = List[Dependency]


class CmCorefIndexer():

    def __init__(self, document: Document, lang: Lang):
        self.lang = lang
        self.document = document

    
    def get_cm_syntactic_graph(self, sentence: Sentence, sentence_index: int) -> CmSyntacticGraph:
        dependencies = sentence.get_dependencies()
        syntactic_graph = CmSyntacticGraph()

        for index, dependency in enumerate(dependencies):
            dependent_word = self.get_actual_word(dependency[0], sentence_index)
            governor_word = self.get_actual_word(dependency[1], sentence_index)

            if dependent_word.is_content_word() and governor_word.is_content_word():
                syntactic_graph.index_edge(dependent_word, governor_word)
                continue

            if not dependent_word.is_content_word() and governor_word.is_content_word():
                content_word = governor_word
                non_content_word = dependent_word
            elif dependent_word.is_content_word() and not governor_word.is_content_word():
                content_word = dependent_word
                non_content_word = governor_word
            else:
                continue

            clone = sentence.get_dependencies()
            del clone[index]
            self.explore(content_word, non_content_word, clone, syntactic_graph, sentence_index)

        return syntactic_graph
            

    def explore(self, content_word: Word, non_content_word: Word, dependencies: Dependencies,
                    syntactic_graph: CmSyntacticGraph, sentence_index: int) -> None:
        for index, dependency in enumerate(dependencies):
            dependent_word = self.get_actual_word(dependency[0], sentence_index)
            governor_word = self.get_actual_word(dependency[1], sentence_index)

            if dependent_word.lemma == non_content_word.lemma:
                if governor_word.is_content_word():
                    syntactic_graph.index_edge(content_word, governor_word)
                else:
                    clone = deepcopy(dependencies)
                    clone.remove(index)
                    self.explore(content_word, governor_word, clone, syntactic_graph, sentence_index)
            elif governor_word.lemma == non_content_word.lemma:
                if dependent_word.is_content_word():
                    syntactic_graph.index_edge(content_word, dependent_word)
                else:
                    clone = deepcopy(dependencies)
                    clone.remove(index)
                    self.explore(content_word, dependent_word, clone, syntactic_graph, sentence_index)
                

    # TODO use coref
    def get_actual_word(self, word: Word, sentence_index: int) -> Word:
        return word

    # TODO 
    def index_coreference(self):
        pass