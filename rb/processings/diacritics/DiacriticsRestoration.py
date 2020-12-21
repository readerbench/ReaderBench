import os
import pickle
import sys

import absl
import numpy as np
import rb.processings.diacritics.utils as utils
import tensorflow as tf
from rb.core.lang import Lang
from rb.processings.diacritics.BertCNN import (
    BertCNN, categorical_acc, weighted_categorical_crossentropy)
from rb.processings.diacritics.CharCNN import CharCNN
from rb.processings.encoders.bert import BertWrapper
from rb.utils.downloader import check_version, download_model
from tensorflow.keras.models import load_model


class DiacriticsRestoration(object):
    """
    Wrapper for Diacritics restoration
    """

    def __init__(self, model_name = "base", bert_max_seq_len=128, max_sentence_length=128, max_windows=100, max_sentences=5):

        self.bert_max_seq_len = min(bert_max_seq_len, 512)
        self.max_windows = min(max_windows, 280)
        self.max_sentences = min(max_sentences, 10)
        # load model
        self._load_model(model_name)
        self.max_sentence_length = min(max_sentence_length, 256)

    # loads best diacritics model, i.e CharCNN + RoBERT-base FN
    def _load_model(self, model_name):
        self.bert_wrapper = BertWrapper(Lang.RO, max_seq_len=self.bert_max_seq_len, model_name=model_name, custom_model=True, load_model=True)
        if check_version(Lang.RO, ["models", "diacritice", model_name]):
            download_model(Lang.RO, ["models", "diacritice", model_name])
        self.char_to_id_dict = pickle.load(open(f"resources/ro/models/diacritice/{model_name}/char_dict", "rb"))
        model_path = f"resources/ro/models/diacritice/{model_name}/model.h5"
        
        conv_layers = [[50,2], [50,3], [50,4], [50,5], [50,11]]
        self.model = BertCNN(window_size = 11, alphabet_size = len(self.char_to_id_dict), conv_layers = conv_layers, fc_hidden_size = 128,
					embedding_size = 50, num_of_classes = 5, batch_max_sentences = self.max_sentences, batch_max_windows = self.max_windows,
					bert_wrapper = self.bert_wrapper, bert_trainable = False, cnn_dropout_rate = 0.1, learning_rate = 0, init_model = model_path).model

    # replace_all: replaces all diacritics(if existing) with model predictions
    # replace_missing: replaces only characters that accept and don't have diacritics with model predictions; keeps existing diacritics
    def process_string(self, string, mode="replace_all"):
        full_diacritics = set("aăâiîsștț")
        explicit_diacritics = set("ăâîșțĂÂÎȘȚ")
        if len(string) > self.max_sentence_length:
            result = ""
            for i in range(0, len(string), self.max_sentence_length):
                substring = string[i:min(len(string), i+self.max_sentence_length)]
                result += self.process_string(substring, mode)
            return result
        working_string = string.lower()
        clean_string = ""
        # remove everything not in char_to_id_dict
        for s in working_string:
            if s in self.char_to_id_dict.keys():
                clean_string += s

        working_string = clean_string
        working_string = ''.join([utils.get_char_basic(char) for char in working_string])

        diac_count = 0
        for s in working_string:
            if s in full_diacritics:
                diac_count += 1

        x_dataset = tf.data.Dataset.from_generator(lambda : utils.generator_bert_cnn_features_string(working_string, self.char_to_id_dict, 11, self.bert_wrapper, self.max_sentences, self.max_windows),
                        output_types=({'bert_input_ids': tf.int32, 'bert_segment_ids': tf.int32, 'token_ids': tf.int32, 'sent_ids': tf.int32,
                                        'mask': tf.float32, 'char_windows': tf.int32}, tf.float32),
                        output_shapes=({'bert_input_ids':[self.max_sentences, self.bert_max_seq_len], 'bert_segment_ids':[self.max_sentences, self.bert_max_seq_len], 'token_ids':[self.max_windows],
                                        'sent_ids': [self.max_windows], 'mask': [self.max_windows], 'char_windows': [self.max_windows, 11]}, [self.max_windows, 5]))
        x_dataset = x_dataset.batch(1)

        predictions = self.model.predict(x_dataset, steps=(diac_count//self.max_windows)+1)

        filtered_predictions = []
        for index in range(len(predictions[0])):
            if predictions[1][index] == 1:
                filtered_predictions.append(predictions[0][index])
            
        predictions = np.array(filtered_predictions)
        predicted_classes = list(map(lambda x: np.argmax(x), predictions))
        prediction_index = 0
        
        complete_string = ""
        for orig_char in string:

            lower_orig_char = orig_char.lower()

            if lower_orig_char in full_diacritics:
                if mode == "replace_all":
                    new_char = utils.get_char_from_label(utils.get_char_basic(lower_orig_char), predicted_classes[prediction_index])
                
                elif mode == "replace_missing":					
                    if lower_orig_char in explicit_diacritics:
                        new_char = lower_orig_char
                    else:
                        new_char = utils.get_char_from_label(utils.get_char_basic(lower_orig_char), predicted_classes[prediction_index])
                
                prediction_index += 1

                if orig_char.isupper():
                    new_char = new_char.upper()

            else:
                new_char = orig_char
            complete_string += new_char

        return complete_string
