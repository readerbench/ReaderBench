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
from rb.processings.sentiment.BertRegression import BertRegression

class SentimentAnalysis(object):
	"""
	Wrapper for Sentiment Analysis
    """

	def __init__(self, lang: Lang, model_type="base", max_seq_len=128, check_updates = True):
		# load model
		self.lang = lang
		self.max_seq_len = 128
		self._load_model(model_type, check_updates)

	# loads best sentiment model
	def _load_model(self, model_type, check_updates = True):
		self.bert_wrapper = BertWrapper(self.lang, max_seq_len=self.max_seq_len, model_name=model_type, check_updates=check_updates)
		self.model = BertRegression(bert_wrapper=self.bert_wrapper, bert_trainable=False, bert_pooling_type="cls",
					learning_rate=0, hidden_size=[128,64], restore_model="13",
					optimizer="adam", loss="mse", dropout_rate=0.1, models_path="rb/processings/sentiment/models/").model
		# if check_updates and check_version(Lang.RO, ["models", "sentiment", model_type]):
		# 	download_model(Lang.RO, ["models", "sentiment", model_type])
		# model_path = f"resources/{self.lang.value}/models/sentiment/{model_type}/model.h5"
		# self.model = tf.keras.models.load_model(model_path, custom_objects={
        #                             	'BertModelLayer': self.bert_wrapper.bert_layer, 
		# 							})

	def process_text(self, text):

		if isinstance(text, str):
			text = [text]

		features = utils.processFeaturesRawText(text, self.bert_wrapper)
		predictions = self.model.predict(features)
		processed_predictions = list(map(lambda x: x[0] / 5.0, predictions))
		return processed_predictions
