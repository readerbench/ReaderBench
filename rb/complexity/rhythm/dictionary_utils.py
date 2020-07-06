import re

RE_WHITESPACE = re.compile(r"(\s)+", re.UNICODE)


def merge_and_rewrite():
    path1 = "/home/vacioaca/Downloads/s50_0_words_all.txt"
    path2 = "/home/vacioaca/Downloads/s50_1_words_all.txt"
    dest_path = "/home/vacioaca/Downloads/syllabified_dict.dict"
    dictionary = {}
    with open(path1, mode="r", encoding="utf-8-sig") as file1:
        for line in file1.readlines():
            line = RE_WHITESPACE.sub(" ", line.strip())
            if len(line) == 0:
                continue
            pieces = line.split(" ", 1)
            word = pieces[1].split(" ", 1)[0]
            syllables = pieces[0].split("-")
            if word not in dictionary:
                dictionary[word] = [syllables]
            else:
                filtered_syllables = filter(lambda t: set(t) == set(syllables), dictionary[word])
                if next(filtered_syllables, None) is None:
                    dictionary[word] += [syllables]

    with open(path2, mode="r", encoding="utf-8-sig") as file2:
        for line in file2.readlines():
            line = RE_WHITESPACE.sub(" ", line.strip())
            if len(line) == 0:
                continue
            pieces = line.split(" ", 1)
            word = pieces[1].split(" ", 1)[0]
            syllables = pieces[0].split("-")
            if word not in dictionary:
                dictionary[word] = [syllables]
            else:
                filtered_syllables = filter(lambda t: set(t) == set(syllables), dictionary[word])
                if next(filtered_syllables, None) is None:
                    dictionary[word] += [syllables]

    with open(dest_path, mode="w", encoding="utf-8") as file:
        for key in sorted(dictionary.keys()):
            for syllables in dictionary[key]:
                file.write("{} {}\n".format(key, "-".join(syllables)))
