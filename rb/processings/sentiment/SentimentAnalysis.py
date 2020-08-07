#pylint: disable=import-error
import os
import pickle
import sys
import absl
import numpy as np
import rb.processings.sentiment.utils_new as utils
import tensorflow as tf
from rb.core.lang import Lang
from rb.processings.encoders.bert import BertWrapper
from rb.utils.downloader import check_version, download_model
from tensorflow.keras.models import load_model
import tensorflow.keras as keras

class SentimentAnalysis(object):
	"""
	Wrapper for Sentiment Analysis
    """

	def __init__(self, lang: Lang, model_type="base", check_updates = True):
		# load model
		self.lang = lang
		self.max_seq_len = 128
		self._load_model(model_type, check_updates)

	# loads best sentiment model
	def _load_model(self, model_type, check_updates = True):
		self.bert_wrapper = BertWrapper(self.lang, max_seq_len=self.max_seq_len, model_name=model_type, check_updates=check_updates)
		if check_updates and check_version(Lang.RO, ["models", "sentiment", model_type]):
			download_model(Lang.RO, ["models", "sentiment", model_type])
		model_path = f"resources/{self.lang.value}/models/sentiment/{model_type}"
		self.model = tf.keras.models.load_model(model_path)
		

	def process_text(self, text):

		if isinstance(text, str):
			text = [text]

		features = utils.processFeaturesRawText(text, self.bert_wrapper)
		predictions = self.model.predict(features)
		predictions = list(map(lambda x: x[0], predictions))
		return predictions
