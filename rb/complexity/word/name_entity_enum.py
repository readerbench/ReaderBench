from enum import Enum, auto

class NamedEntityWikiEnum(Enum):
    PER = auto()
    LOC = auto()
    ORG = auto()
    MISC = auto()


class NamedEntityONEnum(Enum):
    PERSON = auto()
    NORP = auto()
    FAC	= auto()
    ORG	= auto()
    GPE	= auto()
    LOC	= auto()
    PRODUCT	= auto()
    EVENT = auto()
    WORK_OF_ART	= auto()
    LAW	= auto()
    LANGUAGE = auto()
    DATE = auto()
    TIME = auto()
    PERCENT	= auto()
    MONEY = auto()
    QUANTITY = auto()
    ORDINAL	= auto()
    CARDINAL = auto()