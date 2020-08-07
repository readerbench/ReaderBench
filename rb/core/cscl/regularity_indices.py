from enum import Enum, unique

@unique
class RegularityIndices(Enum):
    PWD = 'pwd'
    PDH = 'pdh'
    WS1 = 'ws1'
    WS2 = 'ws2'
    WS3 = 'ws3'