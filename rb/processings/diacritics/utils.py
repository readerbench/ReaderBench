import statistics
import random
import sys
import pickle
import tensorflow as tf
import os
from sklearn.utils import class_weight
import numpy as np
from collections import Counter

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
            print("Wrong utils.get_label", basic_char, "->", diacritic_char)

    if basic_char == 'i':
        if diacritic_char == "î":
            return 2
        else:
            print("Wrong utils.get_label", basic_char, "->", diacritic_char)

    if basic_char == 's':
        if diacritic_char == "ș":
            return 3
        else:
            print("Wrong utils.get_label", basic_char, "->", diacritic_char)

    if basic_char == 't':
        if diacritic_char == "ț":
            return 4
        else:
            print("Wrong utils.get_label", basic_char, "->", diacritic_char)

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
# this generator works at sentence level
# for each sentence it returns -> bert_inputs + list of token_idx + list of windows + list of labels
# bert_input_ids :: [bert_max_seq_len]
# bert_segment_ids :: [bert_max_seq_len]
# token_ids :: [no_windows] this is not fixed, it depends on the sentence
# char_windows :: [no_windows, window_size] # this is actually list of numpy arrays
# labels :: [no_windows, no_classes] 
def generator_sentence_bert_cnn_features(filepath, char_to_id_dict, window_size, bert_wrapper):
    
    diacritics = "aăâiîsștț"
    id_to_char_dict = {v: k for k, v in char_to_id_dict.items()}

    with open(filepath, "r", encoding='utf-8') as in_file:
        for _, sentence in enumerate(in_file):

            basic_sentence = ''.join([get_char_basic(char) for char in sentence])
            tokens = bert_wrapper.tokenizer.tokenize(basic_sentence)
            sentence_bert_input_ids, sentence_bert_segment_ids = bert_wrapper.process_text(basic_sentence)
            sentence_token_ids = []
            sentence_char_cnn_windows = []
            sentence_labels = []
            # save map: char_diac_index -> bert_tokenizer_index
            # example: privata -> 'privat' + '##a'
            # 0 (first a) -> 0
            # 1 (second a) -> 1
            sentence_char_index = {}
            
            char_index = 0
            for token_index, token in enumerate(tokens):
                for char in token:
                    if char in diacritics:
                        # offset by 1 because of '[CLS]'
                        sentence_char_index[char_index] = token_index + 1
                        char_index += 1
            
            char_ids = [char_to_id_dict[get_char_basic(char)] for char in sentence]
            values_to_pad = (window_size-1)//2
            char_ids = [char_to_id_dict["<PAD>"]] * values_to_pad + char_ids + [char_to_id_dict["<PAD>"]] * values_to_pad

            char_dia_index = 0
            for char_index, char in enumerate(sentence):
                # we padded vector
                char_index += values_to_pad
                if char in diacritics:

                    full_window = char_ids[(char_index - values_to_pad):(char_index + values_to_pad + 1)]
                    label = get_label(id_to_char_dict[char_ids[char_index]], char)
                    
                    categorical = np.zeros((5))
                    categorical[label] = 1
                    
                    sentence_labels.append(categorical)
                    sentence_token_ids.append(sentence_char_index[char_dia_index])
                    sentence_char_cnn_windows.append(full_window)
                    
                    char_dia_index += 1

                    # print(sentence_bert_input_ids, sentence_bert_segment_ids[char_dia_index], sentence_token_ids, sentence_char_cnn_windows,sentence_labels)
            # sys.exit()
            yield sentence_bert_input_ids, sentence_bert_segment_ids, sentence_token_ids, sentence_char_cnn_windows, sentence_labels
    

# high level generator for bert+cnn   
# output
# bert_input_ids :: [max_sentences, bert_max_seq_len]
# bert_segment_ids :: [max_sentences, bert_max_seq_len]
# token_ids :: [max_windows] for referencing token  
# sent_ids :: [max_windows] for referencing sentence
# mask :: [max_windows] for marking if a window is part of dataset or padding 
# char_windows :: [max_windows, window_size]
# labels :: [max_windows, no_classes]
def generator_bert_cnn_features(filepath, char_to_id_dict, window_size, bert_wrapper, max_sentences, max_windows,):

    padding_window = [0] * window_size
    padding_labels = np.array([0, 0, 0, 0, 0])
    padding_input_ids, padding_segment_ids = bert_wrapper.process_text("")

    crt_sentences = 0
    crt_windows = 0

    bert_input_ids = []
    bert_segment_ids =[]
    token_ids = []
    sentence_ids = []
    windows_mask = []
    char_windows = []
    labels = []

    sentence_generator = generator_sentence_bert_cnn_features(filepath, char_to_id_dict, window_size, bert_wrapper)
    for sentence_entry in sentence_generator:
        sentence_bert_input_ids, sentence_bert_segment_ids, sentence_token_ids, sentence_char_cnn_windows, sentence_labels = sentence_entry
        # print(sentence_bert_input_ids, sentence_bert_segment_ids, sentence_token_ids, sentence_char_cnn_windows, sentence_labels)
        
        bert_input_ids.append(sentence_bert_input_ids)
        bert_segment_ids.append(sentence_bert_segment_ids)

        for window_index in range(len(sentence_token_ids)):
            token_ids.append(sentence_token_ids[window_index])
            sentence_ids.append(crt_sentences)
            windows_mask.append(1.0)
            char_windows.append(sentence_char_cnn_windows[window_index])
            labels.append(sentence_labels[window_index])
            
            crt_windows += 1

            if crt_windows == max_windows:

                sentences_to_pad = max_sentences - crt_sentences - 1
                bert_input_ids = bert_input_ids + [padding_input_ids] * sentences_to_pad
                bert_segment_ids = bert_segment_ids + [padding_segment_ids] * sentences_to_pad
                
                yield {'bert_input_ids':bert_input_ids, 'bert_segment_ids':bert_segment_ids, 'token_ids': token_ids, 
                    'sent_ids': sentence_ids, 'mask': windows_mask, 'char_windows': char_windows}, labels

                # take the last sentence before padding
                bert_input_ids = [bert_input_ids[crt_sentences]]
                bert_segment_ids = [bert_segment_ids[crt_sentences]]
                # reset global vars
                crt_sentences = 0
                crt_windows = 0

                token_ids = []
                sentence_ids = []
                windows_mask = []
                char_windows = []
                labels = []

        crt_sentences += 1
        if crt_sentences == max_sentences:
            # we have reached maximum sentence count
            # we need to pad up to max_window_size
            values_to_pad = max_windows - crt_windows

            token_ids = token_ids + [0] * values_to_pad
            sentence_ids = sentence_ids + [0] * values_to_pad
            windows_mask = windows_mask + [0] * values_to_pad
            char_windows = char_windows + [padding_window] * values_to_pad
            labels = labels + [padding_labels] * values_to_pad

            yield {'bert_input_ids':bert_input_ids, 'bert_segment_ids':bert_segment_ids, 'token_ids': token_ids, 
                    'sent_ids': sentence_ids, 'mask': windows_mask, 'char_windows': char_windows}, labels

            # reset global vars
            crt_sentences = 0
            crt_windows = 0
            bert_input_ids = []
            bert_segment_ids =[]
            token_ids = []
            sentence_ids = []
            windows_mask = []
            char_windows = []
            labels = []

    # return uncompleted
    # we have to pad up to max_sentences and max_windows
    # pad up to max_sentences
    sentences_to_pad = max_sentences - crt_sentences
    bert_input_ids = bert_input_ids + [padding_input_ids] * sentences_to_pad
    bert_segment_ids = bert_segment_ids + [padding_segment_ids] * sentences_to_pad

    # pad up to max_windows
    values_to_pad = max_windows - crt_windows
    token_ids = token_ids + [0] * values_to_pad
    sentence_ids = sentence_ids + [0] * values_to_pad
    windows_mask = windows_mask + [0] * values_to_pad
    char_windows = char_windows + [padding_window] * values_to_pad
    labels = labels + [np.zeros(5)] * values_to_pad
    
    # print("BII", len(bert_input_ids))#, bert_input_ids)
    # print("BSI", len(bert_segment_ids))#, bert_segment_ids)
    # print("Token ids", len(token_ids))#, token_ids)
    # print("Sent ids", len(sentence_ids))#, sentence_ids)
    # print("Window mask", len(windows_mask))#, windows_mask)
    # print("Char windows", len(char_windows))#, char_windows)
    # print("Labels", len(labels))#, labels)

    yield {'bert_input_ids':bert_input_ids, 'bert_segment_ids':bert_segment_ids, 'token_ids': token_ids, 
            'sent_ids': sentence_ids, 'mask': windows_mask, 'char_windows': char_windows}, labels

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


def evaluate_model(model, filepath, dataset, steps, model_type="BertCNN", write_to_file=False, outfile_name=None):

    diacritics = set("aăâiîsștț")
    predictions = model.predict(dataset, steps=steps)
    
    if model_type == "BertCNN":
        filtered_predictions = []
        for index in range(len(predictions[0])):
            if predictions[1][index] == 1:
                filtered_predictions.append(predictions[0][index])
            
        predictions = np.array(filtered_predictions)
    
    predicted_classes = list(map(lambda x: np.argmax(x), predictions))
    print(predictions.shape, len(predicted_classes))
    predicted_dia = []
    predicted_cla = []

    global_true_words = []
    global_predicted_words = []

    global_true_chars = []
    global_predicted_chars = []

    prediction_index = 0

    with open(filepath, "r", encoding='utf-8') as in_file:
        for _, sentence in enumerate(in_file):

            global_true_chars.extend(sentence)
            sentence_true_words = sentence.split(" ")
            global_true_words.extend(sentence_true_words)

            for _, char in enumerate(sentence):
                if char in diacritics:
                    basic_char = get_char_basic(char)
                    predicted_char = get_char_from_label(basic_char, predicted_classes[prediction_index])
                    global_predicted_chars.append(predicted_char)
        
                    predicted_dia.append(predicted_char)
                    predicted_cla.append(predicted_classes[prediction_index])

                    prediction_index += 1

                else: 
                    global_predicted_chars.append(char)

    global_predicted_words = ''.join(global_predicted_chars).replace("\n", "\n ").split(" ")[:-1]

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


    print("Word accuracy dia =", format(word_accuracy_dia, '.6f'))
    print("Word accuracy all =", format(word_accuracy, '.6f'))

    print("Char accuracy dia =", format(char_accuracy_dia, '.6f'))
    print("Char accuracy all =", format(char_accuracy, '.6f'))

    # print(len(predicted_dia), len(predicted_cla))
    print(Counter(predicted_dia), Counter(predicted_cla))

    if write_to_file == True:
        # also write to file
        with open(outfile_name , "w", encoding="utf-8") as outfile:
            for word in global_predicted_words:
                if word[-1] == "\n":
                    outfile.write(word)
                else:
                    outfile.write(word + " ")

    return word_accuracy_dia, word_accuracy, char_accuracy_dia, char_accuracy, global_predicted_words


def generator_sentence_bert_cnn_features_string(string, char_to_id_dict, window_size, tokenizer):
    
    diacritics = "aăâiîsștț"
    id_to_char_dict = {v: k for k, v in char_to_id_dict.items()}

    sentence = string

    basic_sentence = ''.join([get_char_basic(char) for char in sentence])

    tokens = tokenizer.tokenize(basic_sentence)
    features = tokenizer(basic_sentence, return_tensors="tf", padding="max_length", max_length=512)

    sentence_bert_input_ids = features["input_ids"].numpy().tolist()[0]
    sentence_bert_segment_ids = features["token_type_ids"].numpy().tolist()[0]

    sentence_token_ids = []
    sentence_char_cnn_windows = []
    sentence_labels = []
    # save map: char_diac_index -> bert_tokenizer_index
    # example: privata -> 'privat' + '##a'
    # 0 (first a) -> 0
    # 1 (second a) -> 1
    sentence_char_index = {}
    
    char_index = 0
    for token_index, token in enumerate(tokens):
        for char in token:
            if char in diacritics:
                # offset by 1 because of '[CLS]'
                sentence_char_index[char_index] = token_index + 1
                char_index += 1
    
    char_ids = [char_to_id_dict[get_char_basic(char)] for char in sentence]
    values_to_pad = (window_size-1)//2
    char_ids = [char_to_id_dict["<PAD>"]] * values_to_pad + char_ids + [char_to_id_dict["<PAD>"]] * values_to_pad

    char_dia_index = 0
    for char_index, char in enumerate(sentence):
        # we padded vector
        char_index += values_to_pad
        if char in diacritics:

            full_window = char_ids[(char_index - values_to_pad):(char_index + values_to_pad + 1)]
            label = get_label(id_to_char_dict[char_ids[char_index]], char)
            
            categorical = np.zeros((5))
            categorical[label] = 1
            
            sentence_labels.append(categorical)
            sentence_token_ids.append(sentence_char_index[char_dia_index])
            sentence_char_cnn_windows.append(full_window)
            
            char_dia_index += 1

            # print(sentence_bert_input_ids, sentence_bert_segment_ids[char_dia_index], sentence_token_ids, sentence_char_cnn_windows,sentence_labels)
    # sys.exit()
    yield sentence_bert_input_ids, sentence_bert_segment_ids, sentence_token_ids, sentence_char_cnn_windows, sentence_labels


def generator_bert_cnn_features_string(string, char_to_id_dict, window_size, tokenizer, max_sentences, max_windows):

    padding_window = [0] * window_size
    padding_labels = np.array([0, 0, 0, 0, 0])

    features = tokenizer("", return_tensors="tf", padding="max_length", max_length=512)

    padding_input_ids = features["input_ids"].numpy().tolist()[0]
    padding_segment_ids = features["token_type_ids"].numpy().tolist()[0]

    crt_sentences = 0
    crt_windows = 0

    bert_input_ids = []
    bert_segment_ids =[]
    token_ids = []
    sentence_ids = []
    windows_mask = []
    char_windows = []
    labels = []

    sentence_generator = generator_sentence_bert_cnn_features_string(string, char_to_id_dict, window_size, tokenizer)
    for sentence_entry in sentence_generator:
        sentence_bert_input_ids, sentence_bert_segment_ids, sentence_token_ids, sentence_char_cnn_windows, sentence_labels = sentence_entry
        # print(sentence_bert_input_ids, sentence_bert_segment_ids, sentence_token_ids, sentence_char_cnn_windows, sentence_labels)
        
        bert_input_ids.append(sentence_bert_input_ids)
        bert_segment_ids.append(sentence_bert_segment_ids)

        for window_index in range(len(sentence_token_ids)):
            token_ids.append(sentence_token_ids[window_index])
            sentence_ids.append(crt_sentences)
            windows_mask.append(1.0)
            char_windows.append(sentence_char_cnn_windows[window_index])
            labels.append(sentence_labels[window_index])
            
            crt_windows += 1

            if crt_windows == max_windows:

                sentences_to_pad = max_sentences - crt_sentences - 1
                bert_input_ids = bert_input_ids + [padding_input_ids] * sentences_to_pad
                bert_segment_ids = bert_segment_ids + [padding_segment_ids] * sentences_to_pad
                
                yield {'bert_input_ids':bert_input_ids, 'bert_segment_ids':bert_segment_ids, 'token_ids': token_ids, 
                    'sent_ids': sentence_ids, 'mask': windows_mask, 'char_windows': char_windows}, labels

                # take the last sentence before padding
                bert_input_ids = [bert_input_ids[crt_sentences]]
                bert_segment_ids = [bert_segment_ids[crt_sentences]]
                # reset global vars
                crt_sentences = 0
                crt_windows = 0

                token_ids = []
                sentence_ids = []
                windows_mask = []
                char_windows = []
                labels = []

        crt_sentences += 1
        if crt_sentences == max_sentences:
            # we have reached maximum sentence count
            # we need to pad up to max_window_size
            values_to_pad = max_windows - crt_windows

            token_ids = token_ids + [0] * values_to_pad
            sentence_ids = sentence_ids + [0] * values_to_pad
            windows_mask = windows_mask + [0] * values_to_pad
            char_windows = char_windows + [padding_window] * values_to_pad
            labels = labels + [padding_labels] * values_to_pad

            yield {'bert_input_ids':bert_input_ids, 'bert_segment_ids':bert_segment_ids, 'token_ids': token_ids, 
                    'sent_ids': sentence_ids, 'mask': windows_mask, 'char_windows': char_windows}, labels

            # reset global vars
            crt_sentences = 0
            crt_windows = 0
            bert_input_ids = []
            bert_segment_ids =[]
            token_ids = []
            sentence_ids = []
            windows_mask = []
            char_windows = []
            labels = []

    # return uncompleted
    # we have to pad up to max_sentences and max_windows
    # pad up to max_sentences
    sentences_to_pad = max_sentences - crt_sentences
    bert_input_ids = bert_input_ids + [padding_input_ids] * sentences_to_pad
    bert_segment_ids = bert_segment_ids + [padding_segment_ids] * sentences_to_pad

    # pad up to max_windows
    values_to_pad = max_windows - crt_windows
    token_ids = token_ids + [0] * values_to_pad
    sentence_ids = sentence_ids + [0] * values_to_pad
    windows_mask = windows_mask + [0] * values_to_pad
    char_windows = char_windows + [padding_window] * values_to_pad
    labels = labels + [np.zeros(5)] * values_to_pad
    
    # print("BII", len(bert_input_ids))#, bert_input_ids)
    # print("BSI", len(bert_segment_ids))#, bert_segment_ids)
    # print("Token ids", len(token_ids))#, token_ids)
    # print("Sent ids", len(sentence_ids))#, sentence_ids)
    # print("Window mask", len(windows_mask))#, windows_mask)
    # print("Char windows", len(char_windows))#, char_windows)
    # print("Labels", len(labels))#, labels)

    yield {'bert_input_ids':bert_input_ids, 'bert_segment_ids':bert_segment_ids, 'token_ids': token_ids, 
            'sent_ids': sentence_ids, 'mask': windows_mask, 'char_windows': char_windows}, labels








if __name__ == "__main__":
    print("utils.py")
    # build_char_vocab()

