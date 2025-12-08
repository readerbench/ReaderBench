from enum import Enum, auto

class OverlapType(Enum):
    CONTENT_OVERLAP = auto()
    TOPIC_OVERLAP = auto()
    ARGUMENT_OVERLAP = auto()