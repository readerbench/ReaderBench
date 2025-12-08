#pylint: disable=import-error

import numpy as np
import tensorflow as tf
from rb.core.lang import Lang
from rb.utils.downloader import check_version, download_model
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification


class SentimentAnalysis(object):
	"""
	Wrapper for Sentiment Analysis
    """

	def __init__(self, lang: Lang, max_seq_len=128, check_updates = True):
		# load model
		self.lang = lang
		self.max_seq_len = min(max_seq_len, 512)
		if self.lang is Lang.RO:
			if check_updates and check_version(Lang.RO, ["models", "sentiment", "base"]):
				download_model(Lang.RO, ["models", "sentiment", "base"])

			self.model = tf.keras.models.load_model("resources/ro/models/sentiment/base")
			self.tokenizer = AutoTokenizer.from_pretrained("readerbench/RoBERT-base")
			self.max_seq_len = 512
		else:
			self.tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment") 
			self.model = TFAutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment", from_pt=True)
	
		
	def process_text(self, text):

		if isinstance(text, str):
			text = [text]

		if self.lang is Lang.RO:
			features = self.tokenizer(text, return_tensors="tf", padding="max_length", max_length=self.max_seq_len)
			del features['attention_mask']
			predictions = self.model(features).numpy()
			return [x[0] / 5.0 for x in predictions]
		else:
			features = self.tokenizer(text)
			features = {key: np.array(value) for key, value in features.items()}
			predictions = tf.nn.softmax(self.model(features).logits).numpy()
			return [sum(i / 5 * p for i, p in enumerate(pred)) for pred in predictions]
		