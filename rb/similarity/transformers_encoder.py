import os
from rb.core.word import Word
from typing import Dict, List, Union

import numpy as np
import tensorflow as tf
from rb import Block, Document
from rb.core.lang import Lang
from rb.core.text_element import TextElement
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
                # name = "roberta-base"
                name = "sentence-transformers/all-distilroberta-v1"
                from_pt=True
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

    def clean_text(self, text: str) -> str:
        return text.replace("’", "'") \
            .replace("“", '"') \
            .replace("”", '"')

    def _create_word_token_dict(self, blocks: List[Block], tokenized: List[str]) -> Dict[Word, List[int]]:
        i = 0
        result = {}
        words = [self.clean_text(word.text) for block in blocks for word in block.get_words()]
        if getattr(self.tokenizer, "do_lower_case", False):
            words = [word.lower() for word in words]
        block_symbols = {s for word in words for s in word}
        tokens = ["".join(s for s in token if s in block_symbols) for token in tokenized]
        current = ""
        for text, word in zip(words, [word for block in blocks for word in block.get_words()]):
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

    def _encode_block(self, blocks: List[Block]):
        text = "\n".join(block.text for block in blocks)
        tokenized = self.tokenizer.tokenize(self.clean_text(text))
        if not tokenized:
            return
        token_index = self._create_word_token_dict(blocks, tokenized)
        start = 0
        n = len(tokenized)
        token_ids = self.tokenizer.convert_tokens_to_ids(tokenized)
        embeddings = []
        while start < n:
            end = min(n, start + self.max_seq_len - 2)
            input_ids = tf.constant([[self.bos] + token_ids[start:end] + [self.eos]])
            attention_mask = tf.ones(input_ids.shape, dtype=np.int32)
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            embeddings.append(outputs.last_hidden_state[0, 1:-1, :].numpy())
            start = end
        embeddings = np.concatenate(embeddings, axis=0)
        for block in blocks:
            for sentence in block.get_sentences():
                for word in sentence.get_words():
                    if word in token_index:
                        word.vectors[self] = Vector(np.mean(embeddings[token_index[word]], axis=0, dtype=np.float64))
                sentence_ids = [
                    i
                    for word in sentence.get_words() 
                    if word in token_index 
                    for i in token_index[word]
                ]
                if sentence_ids:
                    sentence.vectors[self] = Vector(np.mean(embeddings[sentence_ids], axis=0, dtype=np.float64))
            block_ids = [
                i
                for word in block.get_words() 
                if word in token_index 
                for i in token_index[word]
            ]
            if block_ids:
                block.vectors[self] = Vector(np.mean(embeddings[block_ids], axis=0, dtype=np.float64))
        
    def encode(self, document: Document):
        vectors = []
        buffer = []
        for block in document.components:
            if sum(len(x.text) for x in buffer) + len(block.text) < 1000:
                buffer.append(block)
                continue
            try:
                self._encode_block(buffer)
            except Exception as e:
                text = '\n'.join(x.text for x in buffer)
                logger.warning("Invalid characters:\n" + text)
            for x in buffer:
                if self in x.vectors:
                    vectors.append(x.vectors[self].values)
            buffer = [block]
        try:
            self._encode_block(buffer)
        except Exception as e:
            text = '\n'.join(x.text for x in buffer)
            logger.warning("Invalid characters:\n" + text)
        for x in buffer:
            if self in x.vectors:
                vectors.append(x.vectors[self].values)
        if vectors:
            document.vectors[self] = Vector(np.mean(vectors, axis=0))
            
    def most_similar(self, elem: Union[str, TextElement, Vector], 
                    topN: int = 10, threshold: float = None):
        return []