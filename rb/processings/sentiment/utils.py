import json
import sys
import matplotlib.pyplot as plt
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

# get features from data
# input -> data list of json
# sample_majority -> sample or not from majority class
# sample_count -> how many entries to sample from majority class
# set seed -> random seed value
# output -> list of dicts | one entry is a dict with features and labels
def getFeatures(data, use_review_text=True, sample_majority=False, sample_count=0, seed=None):
    
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
            if data_entry['_source']['Review']['ReviewRating'] == 3:
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
            if data_entry['_source']['Review']['ReviewRating'] != 3:
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
