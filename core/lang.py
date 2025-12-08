from enum import Enum, unique


@unique
class Lang(Enum):
    EN = 'en'
    FR = 'fr'
    RO = 'ro'
    ES = 'es'
    DE = 'de'
    RU = 'ru'
    IT = 'it'
    NL = 'nl'
    PT = "pt"

    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_
