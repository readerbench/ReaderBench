from enum import Enum, unique

@unique
class CsclCriteria(Enum):
    AVERAGE = 'average'
    STDEV = 'stdev'
    SLOPE = 'slope'
    ENTROPY = 'entropy'
    UNIFORMITY = 'uniformity'
    PEAK_CHAT_FRAME = 'peak_chat_frame'
    LOCAL_EXTREME = 'local_extreme'
    RECURRENCE_AVERAGE = 'recurrence_average'
    RECURRENCE_STDEV = 'recurrence_stdev'
    
    