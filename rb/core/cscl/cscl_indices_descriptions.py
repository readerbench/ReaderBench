from enum import Enum, unique
 
@unique
class CsclIndicesDescriptions(Enum):
    INTER_ANIMATION_DEGREE = 'Degree of inter-animation'
    SOCIAL_KB = 'Cumulated social KB scores'
    INDEGREE = 'In-degree centrality'
    NO_CONTRIBUTION = 'Contributions'
    CONTRIBUTIONS_SCORE = 'Cumulated contribution scores'
    RHYTHMIC_COEFFICIENT = 'Personal rhythmic coefficient'
    NO_VERBS = 'Verbs'
    FREQ_MAX_RHYTMIC_INDEX = 'Frequency of maximal rhythmic index'
    RHYTHMIC_INDEX = 'Personal rhythmic index'
    PERSONAL_REGULARITY_ENTROPY = 'Personal chat entropy for regularity measure'
    OUTDEGREE = 'Out-degree centrality'
    NO_NOUNS = 'Nouns'
