#pylint: disable=import-error
import os
#     except RuntimeError as e:
#         print(e)
import pickle
import sys

import absl
import numpy as np
import rb.processings.diacritics.utils as utils
import tensorflow as tf
from rb.core.lang import Lang
from rb.processings.diacritics.BertCNN import (
    BertCNN, categorical_acc, weighted_categorical_crossentropy)
from rb.processings.diacritics.CharCNN import CharCNN
from rb.processings.encoders.bert import BertWrapper
from rb.utils.downloader import check_version, download_model
from tensorflow.keras.models import load_model

# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     try:
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)





class DiacriticsRestoration(object):
	"""
	Wrapper for Diacritics restoration
    """

	def __init__(self):
		# load model
		self._load_model()

	# loads best diacritics model, i.e CharCNN + RoBERT-base FN
	def _load_model(self):
		self.bert_wrapper = BertWrapper(Lang.RO, max_seq_len=128, model_name="base")
		if check_version(Lang.RO, ["models", "diacritice", "RoBERT_base+CNN"]):
			download_model(Lang.RO, ["models", "diacritice", "RoBERT_base+CNN"])
		self.char_to_id_dict = pickle.load(open("resources/ro/models/diacritice/RoBERT_base+CNN/char_dict", "rb"))
		model_path = "resources/ro/models/diacritice/RoBERT_base+CNN/model_11_bert1_tr.h5"
		self.model = load_model(model_path, custom_objects={'BertModelLayer': self.bert_wrapper.bert_layer, 'loss':weighted_categorical_crossentropy(np.ones(5), 5).loss, 
   							'categorical_acc': categorical_acc})

	# replace_all: replaces all diacritics(if existing) with model predictions
	# replace_missing: replaces only characters that accept and don't have diacritics with model predictions; keeps existing diacritics
	def process_string(self, string, mode="replace_all"):
		full_diacritics = set("aăâiîsștț")
		explicit_diacritics = set("ăâîșțĂÂÎȘȚ")

		working_string = string.lower()
		clean_string = ""
		# remove everything not in char_to_id_dict
		for s in working_string:
			if s in self.char_to_id_dict.keys():
				clean_string += s

		working_string = clean_string
		working_string = ''.join([utils.get_char_basic(char) for char in working_string])

		diac_count = 0
		for s in working_string:
			if s in full_diacritics:
				diac_count += 1
		# print(diac_count)
		x_dataset = tf.data.Dataset.from_generator(lambda : utils.generator_bert_cnn_features_string(working_string, self.char_to_id_dict, 11, self.bert_wrapper, 10, 280),
						output_types=({'bert_input_ids': tf.int32, 'bert_segment_ids': tf.int32, 'token_ids': tf.int32, 'sent_ids': tf.int32,
										'mask': tf.float32, 'char_windows': tf.int32}, tf.float32),
						output_shapes=({'bert_input_ids':[10, 128], 'bert_segment_ids':[10, 128], 'token_ids':[280],
										'sent_ids': [280], 'mask': [280], 'char_windows': [280, 11]}, [280, 5]))
		x_dataset = x_dataset.batch(1)

		predictions = self.model.predict(x_dataset, steps=(diac_count//280)+1)

		# print(len(predictions[0]))
		filtered_predictions = []
		for index in range(len(predictions[0])):
			if predictions[1][index] == 1:
				filtered_predictions.append(predictions[0][index])
			
		predictions = np.array(filtered_predictions)
		predicted_classes = list(map(lambda x: np.argmax(x), predictions))
		# print(predictions.shape, len(predicted_classes), predicted_classes)
		# sys.exit()
		prediction_index = 0
		
		complete_string = ""
		for orig_char in string:

			lower_orig_char = orig_char.lower()

			if lower_orig_char in full_diacritics:
				if mode == "replace_all":
					new_char = utils.get_char_from_label(utils.get_char_basic(lower_orig_char), predicted_classes[prediction_index])
				
				elif mode == "replace_missing":					
					if lower_orig_char in explicit_diacritics:
						new_char = lower_orig_char
					else:
						new_char = utils.get_char_from_label(utils.get_char_basic(lower_orig_char), predicted_classes[prediction_index])
				
				prediction_index += 1

				if orig_char.isupper():
					new_char = new_char.upper()

			else:
				new_char = orig_char
			complete_string += new_char

		return complete_string
