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

    text_elements = [TextElementType.SENT.value, TextElementType.BLOCK.value, TextElementType.DOC.value]
    measure_functions = [MeasureFunction.AVG, MeasureFunction.STDEV, MeasureFunction.MAX]
    classes = [NoWord, NoUniqueWordsIndex, NoSentences, NoPunctuations, NoCommas, WdEntropy]
    
    for index_class in classes:
        for text_element in text_elements:
            if index_class is NoSentences and text_element == TextElementType.SENT.value:
                continue
            for measure_function in measure_functions:                
                indices.append(index_class(lang, text_element, measure_function))
    
    indices.append(ChNgramEntropy(lang, ChNgramEntropyEnum.TWO, TextElementType.WORD.value, MeasureFunction.AVG))
    indices.append(ChNgramEntropy(lang, ChNgramEntropyEnum.TWO, TextElementType.WORD.value, MeasureFunction.STDEV))
    indices.append(ChNgramEntropy(lang, ChNgramEntropyEnum.TWO, TextElementType.WORD.value, MeasureFunction.MAX))
    
    # if lang is Lang.RO:
    #     #indices.append(NoCacophonies(Lang.RO, TextElementType.SENT.value, MeasureFunction.AVG))
    #     indices.append(NoCommonErrors(Lang.RO, TextElementType.SENT.value, MeasureFunction.AVG))
    
    return indices

    