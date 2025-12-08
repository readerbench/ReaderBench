from collections import defaultdict

from rb.core.lang import Lang
from rb.utils.downloader import download_syllabified_dict
from rb.utils.rblogger import Logger

logger = Logger.get_logger()


class SyllabifiedDict(defaultdict):
    # Here will be the instances for every language stored.
    __instances = {}

    @staticmethod
    def get_instance(lang: Lang):
        if lang not in SyllabifiedDict.__instances or SyllabifiedDict.__instances[lang] is None:
            SyllabifiedDict(lang)
        return SyllabifiedDict.__instances[lang]

    def __init__(self, lang: Lang):
        if lang in SyllabifiedDict.__instances and SyllabifiedDict.__instances[lang] is not None:
            raise Exception("This class is a singleton!")
        else:
            defaultdict.__init__(self, list)
            if not download_syllabified_dict(lang):
                logger.error("Could not download syllabified dictionary for lang {}".format(lang.value))
                SyllabifiedDict.__instances[lang] = None
                return
            self.dict_filename = "resources/{}/dict/syllabified_dict.dict".format(lang.value)
            pairs = self.entries(lang)
            for key, value in pairs:
                self[key].append(value)
            SyllabifiedDict.__instances[lang] = self

    def entries(self, lang: Lang):
        with open(self.dict_filename, "r") as f:
            for line in f.readlines():
                if len(line) == 0:
                    continue
                if lang == Lang.RO:
                    components = line.strip().split(" ", 2)
                else:
                    components = line.strip().split(" ", 1)
                word = components[0].lower()
                syllables = components[1]
                # yet not supported
                obs = components[2].lower() if len(components) > 2 else None

                syllables_list = []
                if " " in syllables:
                    for syllable in syllables.split("-"):
                        # Note: for some dictionaries syllables are represented using phonemes
                        phonemes = syllable.split()
                        syllables_list.append(list(map(lambda t: t.lower(), phonemes)))
                else:
                    syllables_list = list(map(lambda t: t.lower(), syllables.split("-")))

                yield ((word, syllables_list))


if __name__ == "__main__":
    syllabified_en_dict = SyllabifiedDict.get_instance(Lang.EN)
    print(syllabified_en_dict["brusett"])
    print(syllabified_en_dict["home"])
    syllabified_ro_dict = SyllabifiedDict.get_instance(Lang.RO)
    print(syllabified_ro_dict["soare"])
    print(syllabified_ro_dict["douăsprezece"])
