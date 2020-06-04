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
    NO_NEW_THREADS = 'no_new_threads'
    NEW_THREADS_OVERALL_SCORE = 'new_threads_overall_score'
    NEW_THREADS_CUMULATIVE_SOCIAL_KB = 'new_threads_cumulative_social_kb'
    NEW_THREADS_INTER_ANIMATION_DEGREE = 'new_threads_inter_animation_degree'
    AVERAGE_LENGTH_NEW_THREADS = 'average_length_new_threads'