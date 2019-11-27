from enum import Enum
from typing import List

from rb.complexity.morphology.factory import create as morphology
from rb.complexity.surface.factory import create as surface
from rb.complexity.syntax.factory import create as syntax
from rb.complexity.word.factory import create as word
from rb.complexity.cohesion.factory import create as cohesion
from rb.complexity.rhythm.factory import create as rhythm
from rb.complexity.discourse.factory import create as discourse
from rb.cna.cna_graph import CnaGraph
from rb.core.lang import Lang


class IndexCategory(Enum):
    SURFACE = 0
    MORPHOLOGY = 1
    SYNTAX = 2
    WORD = 3
    COHESION = 4
    RHYTHM = 5
    DISCOURSE = 6

    def create(self, lang: Lang, cna_graph: CnaGraph) -> List["ComplexityIndex"]:
        functions = [surface, morphology, syntax, word, cohesion, rhythm, discourse]
        return functions[self.value](lang, cna_graph)
