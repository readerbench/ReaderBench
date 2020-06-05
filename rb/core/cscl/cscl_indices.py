from enum import Enum, unique

@unique
class CsclIndices(Enum):
    SCORE = 'score'
    SOCIAL_KB = 'social_kb'
    NO_CONTRIBUTION = 'no_contributions'
    OUTDEGREE = 'outdegree'
    INDEGREE = 'indegree'
    NO_NEW_THREADS = 'no_new_threads'
    NEW_THREADS_OVERALL_SCORE = 'new_threads_overall_score'
    NEW_THREADS_CUMULATIVE_SOCIAL_KB = 'new_threads_cumulative_social_kb'
    AVERAGE_LENGTH_NEW_THREADS = 'average_length_new_threads'