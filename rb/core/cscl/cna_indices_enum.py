from enum import Enum, unique

@unique
class CNAIndices(Enum):
    INTERACTION_SCORE = 'interaction_score'
    CONTRIBUTIONS_SCORE = 'contributions_score'
    SOCIAL_KB = 'social_kb'
    OUTDEGREE = 'outdegree'
    INDEGREE = 'indegree'
    CLOSENESS = 'closeness'
    BETWEENNESS = 'betweenness'
    EIGENVECTOR = 'eigenvector'
    NO_NEW_THREADS = 'no_new_threads'
    NEW_THREADS_OVERALL_SCORE = 'new_threads_overall_score'
    NEW_THREADS_CUMULATIVE_SOCIAL_KB = 'new_threads_cumulative_social_kb'
    AVERAGE_LENGTH_NEW_THREADS = 'average_length_new_threads'