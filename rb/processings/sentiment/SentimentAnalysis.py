#pylint: disable=import-error
import os
import pickle
import sys
import absl
import numpy as np
import rb.processings.sentiment.utils as utils
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

	def __init__(self, model_type="base"):
		# load model
		self._load_model(model_type)

	# loads best sentiment model
	def _load_model(self, model_type):
		self.bert_wrapper = BertWrapper(Lang.RO, max_seq_len=512, model_name=model_type)
		model_path = "rb/processings/sentiment/{0}.h5".format(model_type)
		self.model = load_model(model_path, custom_objects={'BertModelLayer': self.bert_wrapper.bert_layer})


	def process_text(self, text):

		if isinstance(text, str):
			text = [text]

		features = utils.processFeatures(text, self.bert_wrapper)
		predictions = self.model.predict(features)
		weight_vector = np.array([0.0, 0.33, 0.66, 1.0])
		scores = []
		for _, pred in enumerate(predictions):
			score = np.sum(pred * weight_vector)
			scores.append(score)
			print(pred, np.argmax(pred))
		return scores
