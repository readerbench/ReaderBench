from enum import Enum
from typing import List

from rb.complexity.morphology.factory import create as morphology
from rb.complexity.surface.factory import create as surface
from rb.complexity.syntax.factory import create as syntax
from rb.complexity.word.factory import create as word
from rb.complexity.cohesion.factory import create as cohesion
from rb.core.lang import Lang


class IndexCategory(Enum):
    SURFACE = 0
    MORPHOLOGY = 1
    SYNTAX = 2
    WORD = 3
    COHESION = 4

    def create(self, lang: Lang) -> List["ComplexityIndex"]:
        functions = [surface, morphology, syntax, word, cohesion]
        return functions[self.value](lang)
