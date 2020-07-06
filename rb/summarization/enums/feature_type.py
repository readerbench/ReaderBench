from enum import Enum, unique


@unique
class FeatureType(Enum):
    BINARY = "binary"
    FREQUENCY = "frequency"
    TFIDF = "tfidf"
