from rb.core.document import Document
from rb.core.lang import Lang

from rb.core.text_element_type import TextElementType
from rb.complexity.measure_function import MeasureFunction

from rb.complexity.rhythm.no_alliterations import NoAlliterations
from rb.complexity.rhythm.no_assonances import NoAssonances

# import nltk
# nltk.download()

txt1 = """
        This is a sample document. 
        It can contain, multiple sentences and paragraphs and repeating sentences.
        This is considered a new block (paragraph).
        Therefore in total are 3 blocks.
    """

txt2 = """
        S-a născut repede la 1 februarie 1852,[3] în satul Haimanale (care astăzi îi poartă numele), fiind primul născut al lui Luca Ștefan Caragiale și al Ecaterinei Chiriac Karaboas. Conform unor surse, familia sa ar fi fost de origine aromână.[6] Tatăl său, Luca (1812 - 1870), și frații acestuia, Costache și Iorgu, s-au născut la Constantinopol, 
        fiind fiii lui Ștefan, un bucătar angajat la sfârșitul anului 1812 de Ioan Vodă Caragea în suita sa.
    """

alliteration1 = """
        Greedy goats gobbled up gooseberries, getting good at grabbing the goodies.
    """
alliteration2 = """
        Up the aisle the moans and screams merged with the sickening smell of woolen black clothes worn in summer weather and green leaves wilting over yellow flowers.
    """
alliteration3 = """
        When the canary keeled over the coal miners left the cave.
    """
alliteration4 = """
        I forgot my flip phone but felt free.
    """

assonance1 = """
        That solitude which suits abstruser musings.
    """
assonance2 = """
        I must confess that in my quest I felt depressed and restless.
    """

# element = Document(Lang.EN, alliteration2)
# index = NoAlliterations(element.lang, TextElementType.SENT.value, None)
# index.process(element)

# element = Document(Lang.EN, assonance1)
# index = NoAssonances(element.lang, TextElementType.SENT.value, None)
# index.process(element)

element = Document(Lang.RO, txt2)
