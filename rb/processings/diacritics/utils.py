import statistics
import random
import sys
import pickle
import tensorflow as tf
import os
from sklearn.utils import class_weight
import numpy as np
from collections import Counter
from bert.tokenization.bert_tokenization import FullTokenizer

# split train file in train and dev
def splitDataset(filepath, train_ratio):
    
    full_sentences = []
    with open(filepath, 'r', encoding='utf-8') as in_file:
        for sentence in in_file:
            full_sentences.append(sentence)


    full_sentences = set(full_sentences)
    sentences = len(full_sentences)
    print("Total sentences =", format(sentences, ",d"))

    dev_count = int((1-train_ratio) * sentences)
    print("Dev sentences =", format(dev_count, ",d"))

    # pick random sentences to be in dev
    random_indexes = random.sample(range(sentences), k=dev_count)
    random_indexes.append(9999999999999)
    random_indexes.sort()


    dev_sentences = []
    train_sentences = []

    random_index = 0
    for sentence_index, sentence in enumerate(full_sentences):
        if sentence_index < random_indexes[random_index]:
            train_sentences.append(sentence)

        elif sentence_index == random_indexes[random_index]:
            random_index += 1
            dev_sentences.append(sentence)
        else:
            print("SHOULD NOT HAPPEN")
            sys.exit()

    print("dev =", format(len(dev_sentences), ",d"), "train =", format(len(train_sentences), ",d"), "total =", format(len(dev_sentences)+len(train_sentences), ",d"))
    inter = list(set(dev_sentences) & set(train_sentences)) 
    print("Intersection =", len(inter))
    
    with open(filepath.split(".")[0]+"_tr.txt", 'w', encoding='utf-8') as train_file:
        train_file.writelines(train_sentences)
    with open(filepath.split(".")[0]+"_de.txt", 'w', encoding='utf-8') as dev_file:
        dev_file.writelines(dev_sentences)

# build char vocab
def build_char_vocab():
    dict_train = statistics.compute_chars_dict("dataset/split/train.txt")
    dict_dev = statistics.compute_chars_dict("dataset/split/dev.txt")
    dict_test = statistics.compute_chars_dict("dataset/split/test.txt")

    full_counter = dict_train + dict_dev + dict_test
    print(full_counter)
    print(len(full_counter))


    char_dict = {}
    char_dict["<PAD>"] = 0
    for char, _ in full_counter.items():
        char_dict[char] = len(char_dict)

    print(char_dict)
    print(len(char_dict))

    pickle.dump(char_dict, open("dataset/split/char_dict", "wb"))

# remove char diacritic
# a -> a
# ă -> a etc
def get_char_basic(char):

    if char == 'a' or char == 'ă' or char == 'â':
        return 'a'
    
    if char == "s" or char == "ș":
        return 's'
    
    if char == "t" or char == "ț":
        return 't'

    if char == "i" or char == "î":
        return 'i'
    
    return char

# get label of transformation from char w/o diacritic to char with diacritic
# no_change -> 0
# a to ă -> 1
# a/i to â/î -> 2
# s to ș -> 3
# t to ț -> 4
def get_label(basic_char, diacritic_char):

    if basic_char == diacritic_char:
        return 0

    if basic_char == 'a':
        if diacritic_char == 'ă':
            return 1
        elif diacritic_char == 'â':
            return 2
        else:
            print("Wrong get_label", basic_char, "->", diacritic_char)

    if basic_char == 'i':
        if diacritic_char == "î":
            return 2
        else:
            print("Wrong get_label", basic_char, "->", diacritic_char)

    if basic_char == 's':
        if diacritic_char == "ș":
            return 3
        else:
            print("Wrong get_label", basic_char, "->", diacritic_char)

    if basic_char == 't':
        if diacritic_char == "ț":
            return 4
        else:
            print("Wrong get_label", basic_char, "->", diacritic_char)

# get predicted char from basic char and predicted class
def get_char_from_label(basic_char, predicted_class):

    if predicted_class == 0:
        return basic_char
    
    if basic_char == 'a' and predicted_class == 1:
        return "ă"
    if basic_char == 'a' and predicted_class == 2:
        return "â"
    
    if basic_char == 'i' and predicted_class == 2:
        return "î"

    if basic_char == 's' and predicted_class == 3:
        return "ș"

    if basic_char == 't' and predicted_class == 4:
        return "ț"
    
    else:
        print("Error in utils.get_char_from_label")
        print("basic_char =", basic_char, "predicted_class =", predicted_class)
        sys.exit()

# generator features for cnn: only window and label
def generator_cnn_features(filepath, char_to_id_dict, window_size):
    diacritics = set("aăâiîsștț")
    # diactritics_ids = list(map(lambda x: char_to_id_dict[x], diacritics))

    id_to_char_dict = {v: k for k, v in char_to_id_dict.items()}

    with open(filepath, "r", encoding='utf-8') as in_file:
        for _, sentence in enumerate(in_file):
            
            char_ids = [char_to_id_dict[get_char_basic(char)] for char in sentence]
            values_to_pad = (window_size-1)//2
            char_ids = [char_to_id_dict["<PAD>"]] * values_to_pad + char_ids + [char_to_id_dict["<PAD>"]] * values_to_pad
            
            for char_index, char in enumerate(sentence):
                # we padded vector
                char_index += values_to_pad
                if char in diacritics:
                    full_window = char_ids[(char_index - values_to_pad):(char_index + values_to_pad + 1)]
                    label = get_label(id_to_char_dict[char_ids[char_index]], char)
                    
                    categorical = np.zeros((5))
                    categorical[label] = 1
                    yield np.array(full_window), categorical

# generator features for bert+cnn: bert_tokens, window, position (relative to bert_tokens) and label
def generator_bert_cnn_features(filepath, char_to_id_dict, window_size, bert_tokenizer):
    
    diacritics = "aăâiîsștț"
    # diactritics_ids = list(map(lambda x: char_to_id_dict[x], diacritics))
    id_to_char_dict = {v: k for k, v in char_to_id_dict.items()}

    with open(filepath, "r", encoding='utf-8') as in_file:
        for _, sentence in enumerate(in_file):

            print(sentence)
            # basic_sentence = list(map(lambda x: get_char_basic(x), sentence))
            # print(basic_sentence)

            tokens = bert_tokenizer.tokenize(sentence)
            input_ids = bert_tokenizer.convert_tokens_to_ids(tokens)
            # save map: char_diac_index -> bert_tokenizer_index
            # example: privata -> 'privat' + '##a'
            # 0 (first a) -> 0
            # 1 (second a) -> 1

            sentence_char_index = {}

            # print(tokens)
            # print(input_ids)

            char_index = 0
            for token_index, token in enumerate(tokens):
                for char in token:
                    if char in diacritics:
                        # print(char_index, char, token, token_index)
                        sentence_char_index[char_index] = token_index
                        char_index += 1
            
            # print(sentence_char_index)
            char_ids = list(map(lambda x: char_to_id_dict[x], sentence))
            values_to_pad = (window_size-1)//2
            for _ in range(values_to_pad):
                char_ids.append(char_to_id_dict["<PAD>"])
                char_ids.insert(0, char_to_id_dict["<PAD>"])

            char_dia_index = 0
            for char_index, char in enumerate(sentence):
                # we padded vector
                char_index += values_to_pad
                if char in diacritics:
                    left = char_ids[char_index-values_to_pad:char_index]
                    right = char_ids[char_index+1:char_index+values_to_pad+1]

                    # get middle char without diacritic
                    true_char = char
                    basic_char = get_char_basic(char)
                    basic_char_id = char_to_id_dict[basic_char]
                    
                    # remove diacritics
                    left = list(map(lambda x: char_to_id_dict[get_char_basic(id_to_char_dict[x])], left))
                    right = list(map(lambda x: char_to_id_dict[get_char_basic(id_to_char_dict[x])], right))

                    full_window = left + [basic_char_id] + right
                    label = get_label(basic_char, true_char)
                    
                    categorical = np.zeros((5))
                    categorical[label] = 1
                    
                    # print(input_ids, sentence_char_index[char_dia_index], full_window, categorical)
                    yield input_ids, sentence_char_index[char_dia_index], full_window, categorical
                    char_dia_index += 1
                    
            sys.exit()

# from diacritics site
# word level accuracy on word that accept dia 
def compute_word_accuracy_dia(true_words, pred_words):
    is_valid = lambda word: any(c in word for c in 'aăâiîsștț')
    n_correct = sum(t == p for t, p in zip(true_words, pred_words) if is_valid(t))
    n_total = sum(is_valid(t) for t in true_words)
    return n_correct / n_total

# word level accuracy on all words
def compute_word_accuracy(true_words, pred_words):
    is_valid = lambda word: True
    n_correct = sum(t == p for t, p in zip(true_words, pred_words) if is_valid(t))
    n_total = sum(is_valid(t) for t in true_words)
    return n_correct / n_total

# word level accuracy on char that accept dia 
def compute_char_accuracy_dia(true_chars, pred_chars):
    is_valid = lambda char: char in 'aăâiîsștț'
    n_correct = sum(t == p for t, p in zip(true_chars, pred_chars) if is_valid(t))
    n_total = sum(is_valid(t) for t in true_chars)
    return n_correct / n_total

# char level accuracy on all chars
def compute_char_accuracy(true_chars, pred_chars):
    is_valid = lambda word: True
    n_correct = sum(t == p for t, p in zip(true_chars, pred_chars) if is_valid(t))
    n_total = sum(is_valid(t) for t in true_chars)
    return n_correct / n_total


def evaluate_model_on_file(model, filepath, char_to_id_dict, window_size):
        
    diacritics = "aăâiîsștț"

    global_true_words = []
    global_predicted_words = []

    global_true_chars = []
    global_predicted_chars = []

    predicted_chars = []

    predicted_cla = []
    predicted_dia = []

    with open(filepath, "r", encoding='utf-8') as in_file:
        sentence_windows = []
        predicted_sentence = []
        predicted_indexes = []
        basic_chars = []
        for sentence_index, sentence in enumerate(in_file):

            global_true_chars.extend(sentence)
            sentence_true_words = sentence.split(" ")
            global_true_words.extend(sentence_true_words)

            # bring chars to base form
            basic_sentence = ''.join(list(map(lambda x: get_char_basic(x), sentence)))
            char_ids = list(map(lambda x: char_to_id_dict[x], basic_sentence))

            values_to_pad = (window_size-1)//2
            for _ in range(values_to_pad):
                char_ids.append(char_to_id_dict["<PAD>"])
                char_ids.insert(0, char_to_id_dict["<PAD>"])

            for char_index, char in enumerate(sentence):
                char_index += values_to_pad
                if char not in diacritics:
                    predicted_sentence.append(char)
                    predicted_chars.append(char)
                else:
                    # generate window
                    left = char_ids[char_index-values_to_pad:char_index]
                    right = char_ids[char_index+1:char_index+values_to_pad+1]
                    full_window = left + [char_to_id_dict[get_char_basic(char)]] + right
                    
                    sentence_windows.append(full_window)

                    predicted_indexes.append(len(predicted_sentence))
                    predicted_sentence.append("X")
                    predicted_chars.append("X")
                    basic_chars.append(get_char_basic(char))
            

            if sentence_index % 1e4 == 0 and sentence_index != 0 :

                prediction_vectors = model.predict(sentence_windows)
                for index, prediction_vector in enumerate(prediction_vectors):
                    predicted_class = np.argmax(prediction_vector)
                    
                    predicted_char = get_char_from_label(basic_chars[index], predicted_class)
                    predicted_chars[predicted_indexes[index]] = predicted_char
                    
                    predicted_dia.append(predicted_char)
                    predicted_cla.append(predicted_class)

                predicted_sentence = ''.join(predicted_chars).replace("\n", "\n ").split(" ")[:-1]

                global_predicted_words.extend(predicted_sentence)
                global_predicted_chars.extend(predicted_chars)

                # print(sentence_index, len(sentence_windows))
                # print(global_true_words)
                # print(global_predicted_words)
                
                sentence_windows = []
                predicted_sentence = []
                predicted_indexes = []
                predicted_chars = []
                basic_chars = []

                
            if sentence_index == 15:
                break


    if sentence_windows != []:
        prediction_vectors = model.predict(sentence_windows)
        for index, prediction_vector in enumerate(prediction_vectors):
            predicted_class = np.argmax(prediction_vector)
            
            predicted_char = get_char_from_label(basic_chars[index], predicted_class)
            predicted_chars[predicted_indexes[index]] = predicted_char
            
            predicted_dia.append(predicted_char)
            predicted_cla.append(predicted_class)

        predicted_sentence = ''.join(predicted_chars).replace("\n", "\n ").split(" ")[:-1]

        global_predicted_words.extend(predicted_sentence)
        global_predicted_chars.extend(predicted_chars)



    if len(global_true_words) != len(global_predicted_words):
        print("Mismatch between #true_words and #predicted_words")
        print(len(global_true_words), len(global_predicted_words))
        sys.exit()


    if len(global_predicted_chars) != len(global_predicted_chars):
        print("Mismatch between #true_chars and #predicted_chars")
        print(len(global_true_chars), len(global_predicted_chars))
        sys.exit()
    
    word_accuracy_dia = compute_word_accuracy_dia(global_true_words, global_predicted_words)
    word_accuracy = compute_word_accuracy(global_true_words, global_predicted_words)

    char_accuracy_dia = compute_char_accuracy_dia(global_true_chars, global_predicted_chars)
    char_accuracy = compute_char_accuracy(global_true_chars, global_predicted_chars)


    print("Word accuracy dia =", format(word_accuracy_dia, '.4f'))
    print("Word accuracy all =", format(word_accuracy, '.4f'))

    print("Char accuracy dia =", format(char_accuracy_dia, '.4f'))
    print("Char accuracy all =", format(char_accuracy, '.4f'))

    # print(len(predicted_dia), len(predicted_cla))
    print(Counter(predicted_dia), Counter(predicted_cla))

    return word_accuracy_dia, word_accuracy, char_accuracy_dia, char_accuracy, global_predicted_words

if __name__ == "__main__":
    print("utils.py")
    # build_char_vocab()