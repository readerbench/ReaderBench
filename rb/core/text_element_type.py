from enum import Enum, unique

@unique
class TextElementType(Enum):
    WORD = 0
    SPAN = 1
    SENT = 2
    BLOCK = 3
    DOC = 4
    CONV = 5
    COMM = 6