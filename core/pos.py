from enum import Enum

class POS(Enum):
    ADJ = 'ADJ'
    ADP = 'ADP'
    ADV = 'ADV'
    AUX = 'AUX'
    CONJ = 'CONJ'
    CCONJ = 'CCONJ'
    DET = 'DET'
    INTJ = 'INTJ'
    NOUN = 'NOUN'
    NUM = 'NUM'
    PART = 'PART'
    PRON = 'PRON'
    PROPN = 'PROPN'
    PUNCT = 'PUNCT'
    SCONJ = 'SCONJ'
    SYM = 'SYM'
    VERB = 'VERB'
    X = 'X'
    SPACE = 'SPACE'

    def to_wordnet(self) -> str:
        if self is self.ADJ:
            return 'a'
        if self is self.ADV:
            return 'r'
        if self is self.NOUN:
            return 'n'
        if self is self.VERB:
            return 'v'
        return None
