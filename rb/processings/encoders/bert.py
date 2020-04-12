import os
from typing import List

import bert
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from bert.tokenization import FullTokenizer
from rb.core.lang import Lang
from rb.utils.downloader import check_version, download_model
from tensorflow import keras


class BertWrapper:

    def __init__(self, lang: Lang, model_name: str = None, max_seq_len: int = 256):
        self.lang = lang
        if lang is Lang.EN:
            if model_name is None:
                model_name = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1"
            self.bert_layer = hub.KerasLayer(model_name, trainable=True)
            vocab_file = self.bert_layer.resolved_object.vocab_file.asset_path.numpy()
            do_lower_case = self.bert_layer.resolved_object.do_lower_case.numpy()
            self.tokenizer = FullTokenizer(vocab_file, do_lower_case)
        elif lang is Lang.RO:
            if model_name is None:
                model_name = "ro0"
            self.model_dir = os.path.join("resources/ro/bert/", model_name)
            self.tokenizer = FullTokenizer(vocab_file=os.path.join(model_dir, "vocab.vocab"))
            bert_params = bert.params_from_pretrained_ckpt(self.model_dir)
            self.bert_layer = bert.BertModelLayer.from_params(bert_params, name="bert_layer")
        
        self.max_seq_len = max_seq_len
        
        
    def load_weights(self):
        if self.lang is Lang.RO:
            bert.load_bert_weights(self.bert_layer, os.path.join(self.model_dir, "bert_model.ckpt"))

    def create_inputs(self) -> List[keras.layers.Layer]:
        input_ids = tf.keras.layers.Input(shape=(self.max_seq_len,), dtype=tf.int32, name="input_ids")
        mask_ids = tf.keras.layers.Input(shape=(self.max_seq_len,), dtype=tf.int32, name="mask_ids")
        segment_ids = tf.keras.layers.Input(shape=(self.max_seq_len,), dtype=tf.int32, name="segment_ids")
        return [input_ids, mask_ids, segment_ids]

    def get_output(self, bert_tensor: tf.Tensor, mode: str = "cls") -> tf.Tensor:
        if self.lang is Lang.EN:
            sequence_output = bert_tensor[1]
        elif self.lang is Lang.RO:
            sequence_output = bert_tensor
        if mode == "cls":
            return tf.keras.layers.Lambda(lambda x: x[:,0,:])(sequence_output)
        elif mode == "pool":
            return tf.keras.layers.GlobalAveragePooling1D()(sequence_output)
        return None

    def __get_segments(self, tokens):
        sep_enc = self.tokenizer.convert_tokens_to_ids(['[SEP]'])
        segments = []
        current_segment_id = 0
        for token in tokens:
            segments.append(current_segment_id)
            if token == sep_enc[0]:
                if current_segment_id == 0:
                    current_segment_id = 1
        return segments

    # use it with sentence2=None for single sentence
    def process_sentences(self, sentence1: str, sentence2: str=None):
        tokens = ['[CLS]']
        tokens.extend(self.tokenizer.tokenize(sentence1))
        tokens.append('[SEP]')
        if sentence2 != None:
            tokens.extend(self.tokenizer.tokenize(sentence2))
            tokens.append('[SEP]')

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_masks = [1] * len(input_ids)
        input_segments = self.__get_segments(input_ids)

        if self.max_seq_len == None:
            input_ids = np.array(input_ids)
            input_masks = np.array(input_masks)
            input_segments = np.array(input_segments)
            
            return input_ids, input_segments

        else: # pad or trim
            if len(input_ids) < self.max_seq_len: # pad
                to_add = self.max_seq_len-len(input_ids)
                for _ in range(to_add):
                    input_ids.append(0)
                    input_masks.append(0)
                    input_segments.append(0)

            elif len(input_ids) > self.max_seq_len: # trim
                input_ids = input_ids[:self.max_seq_len]
                input_masks = input_masks[:self.max_seq_len]
                input_segments = input_segments[:self.max_seq_len]

            input_ids = np.array(input_ids)
            input_masks = np.array(input_masks)
            input_segments = np.array(input_segments)

            return input_ids, input_masks, input_segments

# only used for testing purposes
def test_implementation():

    max_seq_length = 10
    if check_version(Lang.RO, ["bert", "ro0"]):
        download_model(Lang.RO, ["bert", "ro0"])
    bert_model = BertLayer("resources/ro/bert/ro0/", max_seq_length)
    
    input_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name="input_ids")
    segment_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name="segment_ids")
    
    bert_output = bert_model([input_ids, segment_ids])
    print(bert_output.shape)

    model = keras.Model(inputs=[input_ids, segment_ids], outputs=bert_output)
    # model.build(input_shape=[(None, bert_model.max_seq_len), (None, bert_model.max_seq_len)])
    # load pretrained
    # bert.load_bert_weights(bert_model.bert_layer, "resources/ro/bert/ro0/bert_model.ckpt")

    # cls_output = keras.layers.Lambda(lambda seq: seq[:, 0, :])(bert_output)
    # logits = keras.layers.Dense(units=1, activation="softmax")(cls_output)

    s1 = "dar se pare ca nu"
    s2 = "how are you?"

    ids, seg = bert_model.process_sentences(sentence1=s1, sentence2=None)
    # print(ids)
    a = model([np.array([ids]), np.array([seg])])
    print(a)
    bert_model.load_weights()
    a = model([np.array([ids]), np.array([seg])])
    print(a)
