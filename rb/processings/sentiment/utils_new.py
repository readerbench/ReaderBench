import json
import sys
# import matplotlib.pyplot as plt
import copy
import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils import class_weight
from collections import Counter
import random
from tensorflow.keras.callbacks import Callback
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from nltk.tokenize import sent_tokenize
import os

# read entire json file
# if loading the original dataset ignore the reviews with score 0
# return a list of json entries
def readJson(file_path, original=False):
    data = []
    with open(file_path, encoding="utf8") as json_file:
        for line in json_file:
            entry = json.loads(line)
            if original == True:
                if entry['_source']['Review']['ReviewRating'] == 0:
                    continue
            data.append(entry)
    return data


# compute histogram of review scores
# input -> list of jsons
# output -> dict score -> #reviews
def computeScoreHistogram(data, normalize = False):
    histo = {}

    for entry in data:
        score = entry['_source']['Review']['ReviewRating']
        if score in histo:
            histo[score] += 1
        else:
            histo[score] = 1

    if normalize == True:
        for key, value in histo.items():
            histo[key] = 1.0 * value / len(data)
    print(histo)
    return histo


def computeTextStatistics(data, superior_threshold=None, inferior_threshold=None):
    histo_char = {}
    histo_word = {}
    histo_category = {}

    sup_threshold = 0
    inf_threshold = 0

    for entry in data:
        text = entry['_source']['Review']['ReviewBody']
        category = entry['_source']['Product']['ProductCategory']
        chars = len(text)
        words = len(text.split(" "))


        if superior_threshold != None and words > superior_threshold:
            sup_threshold += 1

        if inferior_threshold != None and words < inferior_threshold:
            inf_threshold += 1
        
        if chars in histo_char:
            histo_char[chars] += 1
        else:
            histo_char[chars] = 1
    
        if words in histo_word:
            histo_word[words] += 1
        else:
            histo_word[words] = 1

        if category in histo_category:
            histo_category[category] += 1
        else:
            histo_category[category] = 1

    return histo_char, histo_word, histo_category, sup_threshold, inf_threshold


def computeDatasetStatistics(data, superior_threshold=None, inferior_threshold=None):

    histo_scores = computeScoreHistogram(data)
    histo_chars, histo_words, histo_category, sup_threshold, inf_threshold = computeTextStatistics(data, superior_threshold, inferior_threshold)

    print("Reviews with number of words over", superior_threshold, "=", sup_threshold, "percentage =", 100.0*sup_threshold/len(data))
    print("Reviews with number of words under", inferior_threshold, "=", inf_threshold, "percentage =", 100.0*inf_threshold/len(data))
    print(histo_category)
    plt.bar(histo_scores.keys(), histo_scores.values(), 1.0, color='g')
    plt.title("Scores")    
    plt.show()

    plt.bar(histo_chars.keys(), histo_chars.values(), 1.0, color='g')        
    plt.title("Chars")
    plt.show()


    plt.bar(histo_words.keys(), histo_words.values(), 1.0, color='g')        
    plt.title("Words")
    plt.show()

# split the dataset in 5 vs ALL -> 1,2,3,4 -> label 0
#                                        5 -> label 1
# input -> dataset list of jsons
# output -> dataset list of jsons
def splitData5vAll(data):
    new_data = copy.deepcopy(data)
    for entry in new_data:
        if entry['_source']['Review']['ReviewRating'] == 5:
            entry['_source']['Review']['ReviewRating'] = 1
        else:
            entry['_source']['Review']['ReviewRating'] = 0
    return new_data


# save the dataset
# input -> dataset list of jsons, filename to save
def saveData(data, filename):
    
    with open(filename, 'w') as outfile:
        for entry in data:
            json.dump(entry, outfile)
            outfile.write("\n")
    
# get features from data
# input -> data list of json
# sample_majority -> sample or not from majority class
# sample_count -> how many entries to sample from majority class
# set seed -> random seed value
# output -> list of dicts | one entry is a dict with features and labels
def getFeatures(data, use_review_text=True, sample_majority=False, sample_count=0, seed=None, majority_class=3):
    
    if sample_majority == False:
        train_list = []
        for data_entry in data:
            train_entry = {}
            if use_review_text == True:
                train_entry['features:review_text'] = data_entry['_source']['Review']['ReviewBody']
            train_entry['label'] = data_entry['_source']['Review']['ReviewRating']
            train_list.append(train_entry)
        return train_list
    
    elif sample_majority == True:

        majority_list = []
        for data_entry in data:
            majority_entry = {}
            if data_entry['_source']['Review']['ReviewRating'] == majority_class:
                if use_review_text == True:
                    majority_entry['features:review_text'] = data_entry['_source']['Review']['ReviewBody']
                majority_entry['label'] = data_entry['_source']['Review']['ReviewRating']
                majority_list.append(majority_entry)

        random.seed(seed)
        sampled_majority_list = random.sample(majority_list, sample_count)
        random.seed()

        train_list = []
        for data_entry in data:
            train_entry = {}
            if data_entry['_source']['Review']['ReviewRating'] != majority_class:
                if use_review_text == True:
                    train_entry['features:review_text'] = data_entry['_source']['Review']['ReviewBody']
                train_entry['label'] = data_entry['_source']['Review']['ReviewRating']
                # train_list.append(train_entry)
                sampled_majority_list.append(train_entry)
        
        # train_list.extend(sampled_majority_list)
        train_list = sampled_majority_list
        return train_list

# get processed features and labels
# input -> features
# output -> list of processed features, list of labels, dict of class_weights
def processFeatures(data, bert_proc):
    features = []
    labels = []

    iids = []
    sids = []
    i = 0
    for entry in data:
        review_text = entry["features:review_text"]
        input_ids, segment_ids = bert_proc.process_text(review_text)
        iids.append(input_ids)
        sids.append(segment_ids)
        labels.append(entry['label'])    
    
    features = [np.array(iids), np.array(sids)]
    class_weights = class_weight.compute_class_weight('balanced', np.unique(labels), labels)
    class_weights = class_weights.astype(np.float32)
    return features, labels, class_weights

# get processed features and labels from texst
# input -> features
# output -> list of processed features, list of labels, dict of class_weights
def processFeaturesRawText(data, bert_proc):
    features = []
    iids = []
    sids = []
    i = 0
    for entry in data:
        review_text = entry
        input_ids, segment_ids = bert_proc.process_text(review_text)
        iids.append(input_ids)
        sids.append(segment_ids)
    
    features = [np.array(iids), np.array(sids)]
    return features

# split data in train dev test split using stratified 
# input -> data
# output -> train, dev, test data
def splitTrainDevTest(data):
    
    train_data = []
    dev_data = []
    test_data = []

    full_indices = np.array(range(len(data)))
    full_classes = np.array(list(map(lambda x: x['_source']['Review']['ReviewRating'], data)))

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1)

    for tr, te in sss.split(full_indices, full_classes):
        aux_train_indexes = tr
        test_indexes = te

    aux_train_data = []    
    for i in test_indexes:
        test_data.append(data[i])
    for i in aux_train_indexes:
        aux_train_data.append(data[i])


    indices = np.array(range(len(aux_train_data)))
    classes = np.array(list(map(lambda x: x['_source']['Review']['ReviewRating'], aux_train_data)))

    sss_ = StratifiedShuffleSplit(n_splits=1, test_size=0.111111)

    for tr, de in sss_.split(indices, classes):
        train_indexes = tr
        dev_indexes = de

    for i in dev_indexes:
        dev_data.append(aux_train_data[i])
    for i in train_indexes:
        train_data.append(aux_train_data[i])


    print(len(train_data), len(dev_data), len(test_data), len(train_data) + len(dev_data) + len(test_data), len(data))
    print(len(list(set(train_indexes) & set(dev_indexes) & set(test_indexes))))

    return train_data, dev_data, test_data

# split the dataset in 4 classes -> 1 -> label 0
#                                   2,3 -> label 1
#                                   4 -> label 2
#                                   5 -> label 3
# input -> dataset list of jsons
# output -> dataset list of jsons
def splitData4Classes(data):
    new_data = copy.deepcopy(data)
    for entry in new_data:
        if entry['_source']['Review']['ReviewRating'] == 1:
            entry['_source']['Review']['ReviewRating'] = 0
        elif entry['_source']['Review']['ReviewRating'] == 2 or entry['_source']['Review']['ReviewRating'] == 3:
            entry['_source']['Review']['ReviewRating'] = 1
        elif entry['_source']['Review']['ReviewRating'] == 4:
            entry['_source']['Review']['ReviewRating'] = 2
        elif entry['_source']['Review']['ReviewRating'] == 5:
            entry['_source']['Review']['ReviewRating'] = 3
    return new_data


class FScoreCallback(Callback):
    def __init__(self, dataset, steps, labels):

        super().__init__()
        self.steps = steps

        self.dataset = dataset
        self.labels_int = []
        for x in labels:
            self.labels_int.append(np.argmax(x))

    def on_test_end(self, epoch, logs={}):
        y_pred = []
        y_true = self.labels_int
        predict_results = self.model.predict(self.dataset, steps=self.steps)
        for prediction in predict_results:
            y_pred.append(np.argmax(prediction))

        print()
        print(classification_report(y_true, y_pred, digits=4))


def compute_parameters(model_folder_path):

    # define input
    input_ids = tf.keras.layers.Input(shape=(64), dtype=tf.int32, name="input_ids")
    segment_ids = tf.keras.layers.Input(shape=(64), dtype=tf.int32, name="segment_ids")

    import BertModel
    import tensorflow.keras as keras
    import bert

    # define model    
    bert_model = BertModel.BertModel(model_folder_path, 64)
    bert_output = bert_model.bert_layer([input_ids, segment_ids])
    cls_output = keras.layers.Lambda(lambda seq: seq[:, 0, :])(bert_output)
    cls_drop = keras.layers.Dropout(0.1)(cls_output)
    fc1 = keras.layers.Dense(units=100, activation="relu")(cls_drop)
    prediction = keras.layers.Dense(units=10, activation="softmax")(fc1)

    # build model
    model = keras.Model(inputs=[input_ids, segment_ids], outputs=prediction)
    model.build(input_shape=[(None, 64), (None, 64)])
    # load pretrained
    bert.load_bert_weights(bert_model.bert_layer, model_folder_path+"bert_model.ckpt")

    model.compile(optimizer=keras.optimizers.Adam(lr=0.1), loss = 'categorical_crossentropy', metrics = [tf.keras.metrics.categorical_accuracy])
    model.summary()

    from tensorflow.python.keras.utils.layer_utils import count_params
    trainable_count = count_params(model.trainable_weights)
    non_trainable_count = count_params(model.non_trainable_weights)

    print(trainable_count/1e6)
    print(non_trainable_count)

    # return model, bert_model


def build_reallife_corpus(model_folder_path):

    new_model_folder_path = "/".join(model_folder_path.split("/")[:-2])
    new_model_folder_path = os.path.join(new_model_folder_path, "reallife")
    
    train_data = readJson(model_folder_path+"train.json")
    train_data = clean_dict(train_data)
    new_train_data = add_last_sentence_to_data(train_data)
    new_train_data_over = perform_oversampling(new_train_data)
    print(len(train_data), len(new_train_data), len(new_train_data_over))
    saveData(new_train_data_over, os.path.join(new_model_folder_path, "train.json"))


    dev_data = readJson(model_folder_path+"dev.json")
    dev_data = clean_dict(dev_data)
    new_dev_data = add_last_sentence_to_data(dev_data)
    new_dev_data_over = perform_oversampling(new_dev_data)
    print(len(dev_data), len(new_dev_data), len(new_dev_data_over))
    saveData(new_dev_data_over, os.path.join(new_model_folder_path, "dev.json"))

    test_data = readJson(model_folder_path+"test.json")
    test_data = clean_dict(test_data)
    new_test_data = add_last_sentence_to_data(test_data)
    new_test_data_over = perform_oversampling(new_test_data)
    print(len(test_data), len(new_test_data), len(new_test_data_over))
    saveData(new_test_data_over, os.path.join(new_model_folder_path, "test.json"))
    

def add_last_sentence_to_data(data):
    new_data = copy.deepcopy(data)
    new_entries = []
    count = 0
    for entry in new_data:
        review_text = entry['_source']['Review']['ReviewBody']
        sentences = sent_tokenize(review_text)
        if len(sentences) > 1:
            # add new entry to dataset
            new_entry = copy.deepcopy(entry)
            new_entry['_source']['Review']['ReviewBody'] = sentences[-1]
            new_entry['_score'] = 2
            new_entries.append(new_entry)
            if entry == new_entry:
                print(entry)
                print(new_entry)
                sys.exit()
            count += 1
    # print(new_entries)
    new_data.extend(new_entries)
    return new_data


def perform_oversampling(data):
    new_data = copy.deepcopy(data)
    new_entries = []
    counter = [0,0,0,0,0]
    for entry in new_data:
        label = entry['_source']['Review']['ReviewRating']
        counter[label-1] += 1
    
    while True:

        random_entry = random.choice(data)
        random_label = random_entry['_source']['Review']['ReviewRating']

        if counter[random_label-1] == counter[-1]:
            continue
        
        else:
            new_entries.append(random_entry)
            counter[random_label-1] += 1
            
            if counter[0] == counter[1] and counter[1] == counter[2] and counter[2] == counter[3] and counter[3] == counter[4]:
                break

    print(counter)
    new_data.extend(new_entries)
    return new_data


def clean_dict(data):
    new_data = copy.deepcopy(data)
    for entry in new_data:
        del entry["_index"]
        del entry["_type"]
        del entry["_id"]
        del entry["_score"]
        del entry["_source"]["Review"]["ReviewTitle"]
        del entry["_source"]["Review"]["ReviewDate"]
        del entry["_source"]["Review"]["ReviewProductVerified"]
        del entry["_source"]["Product"]
    return new_data



if __name__ == "__main__":

    # data = readJson("../Dataset/Reviews/4Classes/train.json")
    # computeDatasetStatistics(data, 32, 32)

    # print("--------------------------DEV--------------------------")
    # data = readJson("../Dataset/Reviews/4Classes/dev.json")
    # computeDatasetStatistics(data, 32, 32)

    # print("--------------------------TEST--------------------------")
    # data = readJson("../Dataset/Reviews/4Classes/test.json")
    # computeDatasetStatistics(data, 32, 32)

    # compute_parameters("../Models/raw/small/clean/trained_512/ro2/")
    # sys.exit()

    # # split data
    # raw = readJson("../Dataset/Reviews/all_reviews.json", original=True)
    # # computeDatasetStatistics(raw, 256, 256)
    # train_data, dev_data, test_data = splitTrainDevTest(raw)
    # saveData(train_data, "../Dataset/Reviews/emag_train.json")
    # saveData(dev_data, "../Dataset/Reviews/emag_dev.json")
    # saveData(test_data, "../Dataset/Reviews/emag_test.json")


    # raw = readJson("../Dataset/Reviews/all_reviews.json", original=True)
    train_data = readJson("../Dataset/Reviews/emag_train.json")
    # computeDatasetStatistics(train_data, 256, 256)
    dev_data = readJson("../Dataset/Reviews/emag_dev.json")
    test_data = readJson("../Dataset/Reviews/emag_test.json")
    
    computeScoreHistogram(train_data, normalize=True)
    split_train = splitData4Classes(train_data)
    computeScoreHistogram(split_train, normalize=True)
    saveData(split_train, "../Dataset/Reviews/4Classes/train.json")

    computeScoreHistogram(dev_data, normalize=True)
    split_dev = splitData4Classes(dev_data)
    computeScoreHistogram(split_dev, normalize=True)
    saveData(split_dev, "../Dataset/Reviews/4Classes/dev.json")

    computeScoreHistogram(test_data, normalize=True)
    split_test = splitData4Classes(test_data)
    computeScoreHistogram(split_test, normalize=True)
    saveData(split_test, "../Dataset/Reviews/4Classes/test.json")


    