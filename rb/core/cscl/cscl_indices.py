from enum import Enum, unique

@unique
class CsclIndices(Enum):
    SCORE = 'scor'
    SOCIAL_KB = 'social_kb'
    NO_CONTRIBUTION = 'no_contribution'
    NO_NOUNS = 'no_nouns'
    NO_VERBS = 'no_verbs'
    OUTDEGREE = 'outdegree'
    INDEGREE = 'indegree'
