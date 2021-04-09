#pylint: disable=import-error
import os
import pickle
import sys

import absl
import numpy as np
import rb.processings.sentiment.utils_new as utils
import tensorflow as tf
import tensorflow.keras as keras
from rb.core.lang import Lang
from rb.processings.encoders.bert import BertWrapper
from rb.processings.sentiment.BertRegression import BertRegression
from rb.utils.downloader import check_version, download_model
from tensorflow.keras.models import load_model
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification


class SentimentAnalysis(object):
	"""
	Wrapper for Sentiment Analysis
    """

	def __init__(self, lang: Lang, model_type="base", max_seq_len=128, check_updates = True):
		# load model
		self.lang = lang
		self.max_seq_len = min(max_seq_len, 512)
		if self.lang is Lang.RO:
			self._load_model(model_type, check_updates)
		else:
			self.tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment") 
			self.model = TFAutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment", from_pt=True)
	
	# loads best sentiment model
	def _load_model(self, model_type, check_updates = True):
		self.bert_wrapper = BertWrapper(self.lang, max_seq_len=self.max_seq_len, model_name=model_type, check_updates=check_updates, custom_model=True)
		if check_updates and check_version(Lang.RO, ["models", "sentiment", model_type]):
			download_model(Lang.RO, ["models", "sentiment", model_type])
		model_path = f"resources/{self.lang.value}/models/sentiment/{model_type}/"
		self.model = BertRegression(bert_wrapper=self.bert_wrapper, bert_trainable=False, bert_pooling_type="cls",
					learning_rate=0, hidden_size=[128,64], restore_model="",
					optimizer="adam", loss="mse", dropout_rate=0.1, models_path=model_path).model
		
	def process_text(self, text):

		if isinstance(text, str):
			text = [text]

		if self.lang is Lang.RO:
			features = utils.processFeaturesRawText(text, self.bert_wrapper)
			predictions = self.model.predict(features)
			return [x[0] / 5.0 for x in predictions]
		else:
			features = self.tokenizer(text)
			features = {key: np.array(value) for key, value in features.items()}
			predictions = tf.nn.softmax(self.model(features).logits).numpy()
			return [sum(i / 5 * p for i, p in enumerate(pred)) for pred in predictions]
		