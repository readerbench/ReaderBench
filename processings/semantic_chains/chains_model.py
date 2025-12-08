from typing import Dict, List, Tuple, Union

import numpy as np
import tensorflow as tf
from networkx import Graph
from networkx.algorithms.components.connected import connected_components
from rb import POS, Document, Lang, Word
from rb.utils.downloader import check_version, download_model
from transformers import AutoConfig, AutoTokenizer, TFAutoModelForMaskedLM


class ChainsModel():

    def __init__(self, lang = Lang.EN, linear=False, check_updates=True):
        if lang is not Lang.EN:
            raise NotImplementedError(f"Lang {lang.value} not available")
        self.lang = lang
        self.linear = linear
        self.model_type = "linear" if linear else "mlp"
        if check_updates and check_version(self.lang, ["models", "chains", self.model_type]):
            download_model(lang, ["models", "chains", self.model_type])
        self.build_model()
        model_name = "bert-base-cased"
        self.layers = 12
        self.heads = 12
        conf = AutoConfig.from_pretrained(model_name, output_attentions=True, output_past=False, output_additional_info=False, output_hidden_states=False)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert_model = TFAutoModelForMaskedLM.from_pretrained(model_name, config=conf)
        self.max_seq_len = 256
    
    def build_model(self):
        input = tf.keras.layers.Input(288, dtype=tf.float32)
        if self.linear:
            output = tf.keras.layers.Dense(1, activation="sigmoid")(input)
        else:
            hidden = tf.keras.layers.Dense(128, activation="relu")(input)
            hidden = tf.keras.layers.Dense(64, activation="relu")(hidden)
            output = tf.keras.layers.Dense(1, activation="sigmoid")(hidden)
        self.model = tf.keras.Model(inputs=[input], outputs=[output])
        self.model.load_weights(f"resources/{self.lang.value}/models/chains/{self.model_type}/{self.model_type}")

    @staticmethod
    def common_prefix(word1: str, word2: str) -> str:
        i = 0
        while i < min(len(word1), len(word2)):
            if word1[i] != word2[i]:
                return word1[:i]
            i += 1
        return word1[:i]

    def create_word_token_dict(self, doc: Document, tokens: List[int]) -> Dict[Word, List[int]]:
        i = 1
        result = {}
        tokens = [self.tokenizer._convert_id_to_token(token).replace("##", "") for token in tokens]
        current = ""
        for word in doc.get_words():
            if len(word.text) == 0:
                continue
            while i < len(tokens) - 1:
                current = self.common_prefix(word.text, tokens[i])
                if len(current) > 0:
                    break
                i += 1
            ids = [i]
            if word.text == current:
                if len(word.text) < len(tokens[i]):
                    tokens[i] = tokens[i][len(current):]
                else:
                    i += 1
            else: 
                i += 1
                while i < len(tokens) and word.text.startswith(current + tokens[i]):
                    ids.append(i)
                    current += tokens[i]
                    i += 1
                current = word.text[len(current):]
                if len(current) > 0:
                    tokens[i] = tokens[i][len(current):]
            if word.pos in {POS.PRON, POS.NOUN, POS.VERB, POS.PROPN} and not word.is_modal_verb():
                result[word] = ids
        return result
    
    def build_similarity_matrix(self, doc: Union[str, Document]) -> Tuple[List[Word], np.ndarray]:
        if isinstance(doc, str):
            doc = Document(self.lang, doc)
        inputs = self.tokenizer(doc.text)
        start = 1
        n = len(inputs["input_ids"]) - 1
        word_dict = self.create_word_token_dict(doc, inputs["input_ids"])
        words = list(word_dict.keys())
        words_index = {w: i for i, w in enumerate(words)}
        word_scores = np.zeros([len(words), len(words), self.layers, self.heads], np.float32)
        while start < n:
            end = min(n, start + self.max_seq_len - 2)
            input_ids = tf.convert_to_tensor(np.array([[101] + inputs["input_ids"][start:end] + [102]]))
            attention_mask = tf.convert_to_tensor(np.ones(input_ids.shape, dtype=np.int32))
            outputs = self.bert_model(input_ids, attention_mask=attention_mask)
            words_in_batch = [
                (word, [id - start for id in ids if start <= id < end]) 
                for word, ids in word_dict.items() 
                if any(id for id in ids if start <= id < end)
            ]
            scores = np.concatenate(outputs[1], axis=0)[:, :, 1:-1, 1:-1]
            for wb1 in words_in_batch:
                w1 = words_index[wb1[0]]
                for wb2 in words_in_batch:
                    score = sum(scores[:, :, id2, id1] for id1 in wb1[1] for id2 in wb2[1]) / len(wb2[1])
                    w2 = words_index[wb2[0]]
                    for l in range(self.layers):
                        for h in range(self.heads):
                            word_scores[w1, w2, l, h] = max(word_scores[w1, w2, l, h], score[l, h])
            start += self.max_seq_len // 2
        all_features = []
        all_pairs = []
        for i in range(len(words) - 1):
            for j in range(i+1, min(i + self.max_seq_len // 2, len(words))):
                features = [word_scores[i, j, l, h] for l in range(self.layers) for h in range(self.heads)]
                features += [word_scores[j, i, l, h] for l in range(self.layers) for h in range(self.heads)]
                if sum(features) != 0:
                    all_features.append(np.array(features))
                    all_pairs.append((i, j))
        pred_scores = self.model.predict(np.array(all_features))     
        scores = np.zeros([len(words), len(words)], dtype=np.float32)
        for pair, score in zip(all_pairs, pred_scores):
            i, j = pair
            scores[i, j] = score[0]
        return words, scores
    
    def build_semantic_chains(self, doc: Union[str, Document], threshold=0.9) -> List[List[Word]]:
        words, similarity = self.build_similarity_matrix(doc)
        graph = Graph()
        for i, word1 in enumerate(words):
            for j in range(i+1, len(words)):
                if similarity[i, j] > threshold:
                    graph.add_edge(word1, words[j])
        
        return [
            list(sorted(component, key=lambda w: w.index_in_doc)) 
            for component in connected_components(graph)
            ]


        
    
        
		