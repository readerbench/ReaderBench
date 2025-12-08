from enum import Enum, auto

class AoeTypeEnum(Enum):
    INV_AVG = auto()
    INV_LIN_REG_SLOPE = auto()
    IND_ABOVE_THRESH = auto()
    IND_POLY_FIT_ABOVE = auto()
    INFL_POINT_POLY = auto()