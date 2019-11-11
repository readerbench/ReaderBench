from typing import List
from rb.core.lang import Lang
from rb.core.text_element_type import TextElementType
from rb.complexity.measure_function import MeasureFunction
from rb.cna.cna_graph import CnaGraph


def create(lang: Lang, cna_graph: CnaGraph) -> List["ComplexityIndex"]:
    from rb.complexity.surface.no_words import NoWord
    from rb.complexity.surface.no_unq_words import NoUniqueWordsIndex
    from rb.complexity.surface.no_commas import NoCommas
    from rb.complexity.surface.no_punctuations import NoPunctuations
    from rb.complexity.surface.no_sentences import NoSentences
    from rb.complexity.surface.wd_entropy import WdEntropy
    from rb.complexity.surface.ch_entropy import ChEntropy
    from rb.complexity.surface.ch_ngram_entropy import ChNgramEntropyEnum, ChNgramEntropy

    indices = []

    indices.append(NoWord(lang, TextElementType.SENT.value, MeasureFunction.AVG))
    indices.append(NoWord(lang, TextElementType.SENT.value, MeasureFunction.STDEV))

    indices.append(NoWord(lang, TextElementType.BLOCK.value, MeasureFunction.AVG))
    indices.append(NoWord(lang, TextElementType.BLOCK.value, MeasureFunction.STDEV))

    indices.append(NoUniqueWordsIndex(lang, TextElementType.SENT.value, MeasureFunction.AVG))
    indices.append(NoUniqueWordsIndex(lang, TextElementType.SENT.value, MeasureFunction.STDEV))

    indices.append(NoSentences(lang, TextElementType.BLOCK.value, MeasureFunction.AVG))
    indices.append(NoSentences(lang, TextElementType.BLOCK.value, MeasureFunction.STDEV))

    indices.append(NoPunctuations(lang, TextElementType.SENT.value, MeasureFunction.AVG))
    indices.append(NoPunctuations(lang, TextElementType.SENT.value, MeasureFunction.STDEV))

    indices.append(NoPunctuations(lang, TextElementType.BLOCK.value, MeasureFunction.AVG))
    indices.append(NoPunctuations(lang, TextElementType.BLOCK.value, MeasureFunction.STDEV))

    indices.append(NoCommas(lang, TextElementType.SENT.value, MeasureFunction.AVG))
    indices.append(NoCommas(lang, TextElementType.SENT.value, MeasureFunction.STDEV))

    indices.append(NoCommas(lang, TextElementType.BLOCK.value, MeasureFunction.AVG))
    indices.append(NoCommas(lang, TextElementType.BLOCK.value, MeasureFunction.STDEV))

    indices.append(WdEntropy(lang, TextElementType.SENT.value, MeasureFunction.AVG))
    indices.append(WdEntropy(lang, TextElementType.SENT.value, MeasureFunction.STDEV))

    indices.append(WdEntropy(lang, TextElementType.BLOCK.value, MeasureFunction.AVG))
    indices.append(WdEntropy(lang, TextElementType.BLOCK.value, MeasureFunction.STDEV))

    indices.append(WdEntropy(lang, TextElementType.DOC.value, MeasureFunction.AVG))
    indices.append(WdEntropy(lang, TextElementType.DOC.value, MeasureFunction.STDEV))

    indices.append(ChEntropy(lang, TextElementType.WORD.value, MeasureFunction.AVG))
    indices.append(ChEntropy(lang, TextElementType.WORD.value, MeasureFunction.STDEV))

    indices.append(ChNgramEntropy(lang, ChNgramEntropyEnum.TWO, TextElementType.WORD.value, MeasureFunction.AVG))
    indices.append(ChNgramEntropy(lang, ChNgramEntropyEnum.TWO, TextElementType.WORD.value, MeasureFunction.STDEV))

    # if lang is Lang.RO:
    #     #indices.append(NoCacophonies(Lang.RO, TextElementType.SENT.value, MeasureFunction.AVG))
    #     indices.append(NoCommonErrors(Lang.RO, TextElementType.SENT.value, MeasureFunction.AVG))
    
    return indices

    