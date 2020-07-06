from enum import Enum, unique


@unique
class SummarizerType(Enum):
    GENSIM = 'gensim'
    SUBMODULAR = 'submodular'
    TEXTRANK = 'textrank'

    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_

