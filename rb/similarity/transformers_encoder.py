import os
from rb.core.word import Word
from typing import Dict, List

import numpy as np
import tensorflow as tf
from rb import Block, Document
from rb.core.lang import Lang
from rb.similarity.vector import Vector
from rb.similarity.vector_model import VectorModel, VectorModelType
from rb.utils.rblogger import Logger
from transformers import AutoConfig, AutoTokenizer, TFAutoModel

logger = Logger.get_logger()


class TransformersEncoder(VectorModel):

    def __init__(self, lang: Lang, name: str = None, max_seq_len=512):
        from_pt = False
        if name is None:
            if lang is Lang.EN:
                name = "roberta-base"
                self.SOW = "Ġ"
            elif lang is Lang.FR:
                name = "camembert-base"
                self.SOW = "▁"
            elif lang is Lang.RO:
                name = "readerbench/RoBERT-base"
                self.SOW = "##"
            elif lang is Lang.RU:
                name = "DeepPavlov/rubert-base-cased"
                from_pt=True
                self.SOW = "##"
        VectorModel.__init__(self, VectorModelType.TRANSFORMER, name=name, lang=lang, size=768)  
        config = AutoConfig.from_pretrained(self.name)
        config.output_hidden_states = True
        self.tokenizer = AutoTokenizer.from_pretrained(self.name)
        self.bert = TFAutoModel.from_pretrained(self.name, config=config, from_pt=from_pt)
        self.max_seq_len = max_seq_len
        self.bos = self.tokenizer.bos_token_id if self.tokenizer.bos_token_id is not None else self.tokenizer.cls_token_id
        self.eos = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else self.tokenizer.sep_token_id

    @staticmethod
    def common_prefix(word1: str, word2: str) -> str:
        i = 0
        while i < min(len(word1), len(word2)):
            if word1[i] != word2[i]:
                return word1[:i]
            i += 1
        return word1[:i]

    def _create_word_token_dict(self, block: Block, tokenized: List[str]) -> Dict[Word, List[int]]:
        i = 0
        result = {}
        tokens = [token.replace(self.SOW, "") for token in tokenized]
        current = ""
        for word in block.get_words():
            text = word.text
            if getattr(self.tokenizer, "do_lower_case", False):
                text = text.lower()
            while not tokens[i]:
                i += 1
            current = self.common_prefix(text, tokens[i])
            if not current:
                continue 
            ids = [i]
            if text == current:
                if len(text) < len(tokens[i]):
                    tokens[i] = tokens[i][len(current):]
                else:
                    i += 1
            else: 
                i += 1
                while i < len(tokens) and text.startswith(current + tokens[i]):
                    ids.append(i)
                    current += tokens[i]
                    i += 1
                current = text[len(current):]
                if len(current) > 0:
                    tokens[i] = tokens[i][len(current):]
            result[word] = ids
        return result

    def _encode_block(self, block: Block):
        tokenized = self.tokenizer.tokenize(block.text)
        token_index = self._create_word_token_dict(block, tokenized)
        start = 0
        n = len(tokenized)
        token_ids = self.tokenizer.convert_tokens_to_ids(tokenized)
        embeddings = []
        while start < n:
            end = min(n, start + self.max_seq_len - 2)
            input_ids = np.array([[self.bos] + token_ids[start:end] + [self.eos]])
            attention_mask = np.ones(input_ids.shape, dtype=np.int32)
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            embeddings.append(outputs.hidden_states[-2][0, 1:-1, :].numpy())
            start = end
        embeddings = np.concatenate(embeddings, axis=0)
        for sentence in block.get_sentences():
            for word in sentence.get_words():
                if word in token_index:
                    word.vectors[self] = Vector(np.mean(embeddings[token_index[word]], axis=0))
            sentence_ids = [
                i
                for word in sentence.get_words() 
                if word in token_index 
                for i in token_index[word]
            ]
            if sentence_ids:
                sentence.vectors[self] = Vector(np.mean(embeddings[sentence_ids], axis=0))
        block.vectors[self] = Vector(np.mean(embeddings, axis=0))

    def encode(self, document: Document):
        vectors = []
        for block in document.get_blocks():
            self._encode_block(block)
            if self in block.vectors:
                vectors.append(block.vectors[self].values)
        if vectors:
            document.vectors[self] = Vector(np.mean(vectors, axis=0))