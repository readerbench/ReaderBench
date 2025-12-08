from enum import Enum, auto

class EdgeType(Enum):
    PART_OF = auto()
    ADJACENT = auto()
    SEMANTIC = auto()
    LEXICAL_OVERLAP = auto()
    COREF = auto()
    EXPLICIT = auto()