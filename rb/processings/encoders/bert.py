import os
from typing import List, Tuple, Iterable, Union

import bert
import numpy as np
import tensorflow as tf
from bert.tokenization.bert_tokenization import FullTokenizer
from rb.core.lang import Lang
from rb.utils.downloader import check_version, download_model
from tensorflow import keras
from transformers import FlaubertTokenizer, TFFlaubertModel, AutoTokenizer, TFAutoModel
import json


class BertWrapper:

    def __init__(self, lang: Lang, model_name: str = None, max_seq_len: int = 256, check_updates = True, custom_model = False, load_model=True):
        self.lang = lang
        self.custom_model = custom_model
        if lang is Lang.EN:
            if model_name is None:
                model_name = "bert-base-uncased"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            if load_model:
                self.bert_layer = TFAutoModel.from_pretrained(model_name)
        elif lang is Lang.RO:
            if model_name is None:
                model_name = "base"
            if self.custom_model:
                if check_updates and check_version(Lang.RO, ["bert", model_name]):
                    download_model(Lang.RO, ["bert", model_name])
                self.model_dir = os.path.join("resources/ro/bert/", model_name)
                json_config_file = os.path.join(self.model_dir, "bert_config.json")
                config = json.load(open(json_config_file, 'r'))
                do_lower_case = bool(config.get('do_lower_case', 0))
                do_remove_accents = bool(config.get('do_remove_accents', 1))
                self.hidden_size = config.get('hidden_size')
                self.tokenizer = FullTokenizer(vocab_file=os.path.join(self.model_dir, "vocab.vocab"), do_lower_case=do_lower_case)
                if do_remove_accents == False:
                    self.tokenizer.basic_tokenizer._run_strip_accents = lambda x:x
                bert_params = bert.params_from_pretrained_ckpt(self.model_dir)
                if load_model:
                    self.bert_layer = bert.BertModelLayer.from_params(bert_params, name="bert_layer")
            else:
                model_name = f"readerbench/RoBERT-{model_name}"
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                if load_model:
                    self.bert_layer = TFAutoModel.from_pretrained(model_name)
        elif lang is Lang.FR:
            self.tokenizer = FlaubertTokenizer.from_pretrained("jplu/tf-flaubert-base-cased")
            if load_model:
                self.bert_layer = TFFlaubertModel.from_pretrained("jplu/tf-flaubert-base-cased")
                self.bert_layer.call = tf.function(self.bert_layer.call)

        
        self.max_seq_len = max_seq_len
        
        
    def load_weights(self):
        if self.custom_model:
            bert.load_bert_weights(self.bert_layer, os.path.join(self.model_dir, "bert_model.ckpt"))

    def create_inputs(self) -> List[keras.layers.Layer]:
        input_ids = tf.keras.layers.Input(shape=(self.max_seq_len,), dtype=tf.int32, name="input_ids")
        segment_ids = tf.keras.layers.Input(shape=(self.max_seq_len,), dtype=tf.int32, name="segment_ids")
        if self.custom_model:
            return [input_ids, segment_ids]
        mask_ids = tf.keras.layers.Input(shape=(self.max_seq_len,), dtype=tf.int32, name="mask_ids")
        return [input_ids, mask_ids, segment_ids]
        

    def create_inputs_and_model(self) -> Tuple[List[keras.layers.Layer], tf.Tensor]:
        inputs = self.create_inputs()
        if self.lang is Lang.FR:
            output = self.bert_layer(inputs=inputs[0], attention_mask=inputs[1], token_type_ids=inputs[2])
        else:
            output = self.bert_layer(inputs)
        return inputs, output

    def get_output(self, bert_tensor: tf.Tensor, mode: str = "cls") -> tf.Tensor:
        if self.custom_model:
            sequence_output = bert_tensor
        else:
            sequence_output = bert_tensor[0]
        
        if mode == "cls":
            return tf.keras.layers.Lambda(lambda x: x[:,0,:])(sequence_output)
        elif mode == "pool":
            return tf.keras.layers.GlobalAveragePooling1D()(sequence_output)
        elif mode == "seq":
            return sequence_output
        else:
            print("Unrecognized mode {}".format(mode))
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

    def process_input(self, dataset: Iterable[Union[Tuple[str, str], str]]) -> List[np.ndarray]:
        return [np.array(x) for x in zip(*[self.process_text(row) for row in dataset])]

    # use it with sentence2=None for single sentence
    def process_text(self, text: Union[str, Tuple[str, str]]):
        if isinstance(text, str):
            sentence1 = text
            sentence2 = None
        else:
            sentence1, sentence2 = text
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
        else: # pad or trim
            if len(input_ids) < self.max_seq_len: # pad
                to_add = self.max_seq_len-len(input_ids)
                for _ in range(to_add):
                    input_ids.append(0)
                    input_masks.append(0)
                    input_segments.append(0)

            elif len(input_ids) > self.max_seq_len: # trim
                if sentence2 is not None:
                    len1 = 0
                    for segment in input_segments:
                        if segment == 1:
                            break
                        len1 += 1
                    len2 = len(input_segments) - len1
                    each = self.max_seq_len // 2
                    if len1 <= each:
                        input_ids = input_ids[:len1] + input_ids[len1:self.max_seq_len - 1] + [input_ids[len1 + len2 - 1]]
                        input_masks = input_masks[:len1] + input_masks[len1:self.max_seq_len - 1] + [input_masks[len1 + len2 - 1]]
                        input_segments = input_segments[:len1] + input_segments[len1:self.max_seq_len - 1] + [input_segments[len1 + len2 - 1]]
                    elif len2 <= each - 1 + self.max_seq_len % 2:
                        input_ids = input_ids[:(self.max_seq_len - len2 - 1)] + input_ids[(len1-1):]
                        input_masks = input_masks[:(self.max_seq_len - len2 - 1)] + input_masks[(len1-1):]
                        input_segments = input_segments[:(self.max_seq_len - len2 - 1)] + input_segments[(len1-1):]
                    else:
                        input_ids = input_ids[:each] + input_ids[(len1 - 1):(len1 + each - 2 + self.max_seq_len % 2)] + [input_ids[-1]]
                        input_masks = input_masks[:each] + input_masks[(len1 - 1):(len1 + each - 2 + self.max_seq_len % 2)] + [input_masks[-1]]
                        input_segments = input_segments[:each] + input_segments[(len1 - 1):(len1 + each - 2 + self.max_seq_len % 2)] + [input_segments[-1]]
                else:
                    input_ids = input_ids[:self.max_seq_len]
                    input_masks = input_masks[:self.max_seq_len]
                    input_segments = input_segments[:self.max_seq_len]

        if self.custom_model:
            return input_ids, input_segments
        else:
            return input_ids, input_masks, input_segments

# only used for testing purposes
def test_implementation():

    max_seq_length = 10
    if check_version(Lang.RO, ["bert", "ro0"]):
        download_model(Lang.RO, ["bert", "ro0"])
    # bert_model = BertLayer("resources/ro/bert/ro0/", max_seq_length)
    bert_model = lambda x:x

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
