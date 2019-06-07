from typing import List
from rb.core.lang import Lang
from rb.core.text_element_type import TextElementType
from rb.complexity.measure_function import MeasureFunction


def create(lang: Lang) -> List["ComplexityIndex"]:
    from rb.complexity.surface.no_words import NoWordsIndex
    from rb.complexity.surface.no_unq_words import NoUniqueWordsIndex
    from rb.complexity.surface.no_commas import NoCommas
    from rb.complexity.surface.no_punctuations import NoPunctuations
    from rb.complexity.surface.no_sentences import NoSentences
    from rb.complexity.surface.wd_entropy import WdEntropy
    from rb.complexity.surface.ch_entropy import ChEntropy
    from rb.complexity.surface.ch_ngram_entropy import ChNgramEntropyEnum, ChNgramEntropy
    from rb.complexity.surface.no_cacophonies import NoCacophonies
    from rb.complexity.surface.no_common_errors import NoCommonErrors

    from rb.complexity.syntax.dep import DepIndex
    from rb.complexity.syntax.dep_enum import DepEnum

    indices = []
    indices.append(NoWordsIndex(lang, TextElementType.SENT.value, MeasureFunction.AVG))
    indices.append(NoUniqueWordsIndex(lang, TextElementType.SENT.value, MeasureFunction.AVG))
    indices.append(NoSentences(lang, TextElementType.SENT.value, MeasureFunction.AVG))
    indices.append(NoPunctuations(lang, TextElementType.SENT.value, MeasureFunction.AVG))
    indices.append(NoCommas(lang, TextElementType.SENT.value, MeasureFunction.AVG))
    indices.append(WdEntropy(lang, TextElementType.SENT.value, MeasureFunction.AVG))
    indices.append(ChEntropy(lang, TextElementType.WORD.value, MeasureFunction.AVG))
    indices.append(ChNgramEntropy(lang, ChNgramEntropyEnum.TWO, TextElementType.WORD.value, MeasureFunction.AVG))
    indices.append(NoCacophonies(Lang.RO, TextElementType.SENT.value, MeasureFunction.AVG))
    indices.append(NoCommonErrors(Lang.RO, TextElementType.SENT.value, MeasureFunction.AVG))
    # for dep_type in DepEnum:
    #     indices.append(DepIndex(lang, dep_type, TextElementType.SENT.value, MeasureFunction.AVG))
    
    return indices

    