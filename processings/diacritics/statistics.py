from collections import Counter
import sys
import re

def compute_sentences(filepath):
	sentences = 0
	with open(filepath, 'r', encoding='utf-8') as in_file:
		for _ in in_file:
			sentences += 1

	return sentences


def compute_words(filepath):
	words = 0
	with open(filepath, 'r', encoding='utf-8') as in_file:
		for line in in_file:
			words += len(line.split(" "))

	return words


def compute_chars(filepath):
	chars = 0
	with open(filepath, 'r', encoding='utf-8') as in_file:
		for line in in_file:
			chars += len(line)
	return chars


def compute_chars_dict(filepath):
	global_counter = Counter()
	with open(filepath, 'r', encoding='utf-8') as in_file:
		for line in in_file:
			global_counter += Counter(line)	
	return global_counter


def compute_chars_diacritics(filepath):
	regexPattern = re.compile('[aăâiîsștț]')
	chars = 0
	with open(filepath, 'r', encoding='utf-8') as in_file:
		for line in in_file:
			listOfmatches = regexPattern.findall(line)
			chars += len(listOfmatches)
	return chars


def compute_chars_no_spaces(filepath):
	chars = 0
	with open(filepath, 'r', encoding='utf-8') as in_file:
		for line in in_file:
			sentence = line.replace(" ", "")
			sentence = sentence.replace("\n", "")
			chars += len(sentence)
	return chars


def compute_statistics(filepath):
	print("Statistics for", filepath.split("/")[-1])

	sent = compute_sentences(filepath)
	words = compute_words(filepath)
	chars = compute_chars(filepath)
	chars_no_space = compute_chars_no_spaces(filepath)
	chars_dict = compute_chars_dict(filepath)
	chars_diacritics = compute_chars_diacritics(filepath)

	print("Sentences =", format(sent, ",d"))
	print("Words =", format(words, ",d"))
	print("Total chars =", format(chars, ",d"))
	print("Chars w/o spaces =", format(chars_no_space, ",d"))
	print("Char with possible diacritics =", format(chars_diacritics, ",d"))
	print("Chars dict =", chars_dict)
	print("Unique chars =", len(chars_dict))
	print("--------------")


if __name__ == "__main__":

	compute_statistics("dataset/split/train.txt")
	compute_statistics("dataset/split/dev.txt")
	compute_statistics("dataset/split/test.txt")