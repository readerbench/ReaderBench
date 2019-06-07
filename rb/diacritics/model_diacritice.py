import argparse
import itertools
import os
import random
import re
import string
import threading
import time
from typing import List

import keras
import nltk
import numpy as np
import tensorflow as tf
import unidecode
from gensim.models.wrappers import FastText as FastTextWrapper
from keras import backend as K
from tensorflow import keras
from rb.parser.spacy_parser import SpacyParser
from rb.core.lang import Lang
from rb.utils.rblogger import Logger
nltk.download('punkt')

logger = Logger.get_logger()

class Diacritics:

	def __init__(self):
		# set verbosity level */
		tf.logging.set_verbosity(tf.logging.FATAL)
		self.model_embeddings = None
		self.dict_lock = threading.Lock()
		self.dict_avg_words = {}

		# the weights of the model will be saved in a folder per each epoch
		self.fast_text = "fastText/wiki.ro"

		self.window_sentence = 15 
		self.window_character = 6 # 2 * x + 1

		self.character_embedding_size = 28
		self.tag_embedding_size = 5
		self.dep_embedding_size = 5
		self.word_embedding_size = 300


		self.characters_cell_size = 64
		self.sentence_cell_size = 300
		self.neurons_dense_layer_after_merge = 512
		self.batch_size = 128
		self.limit_backtracking_characters = 10

		self.CPUS = 1
		self.size_prefetch_buffer = 10
		self.max_unicode_allowed = 770
		self.replace_character = 255
		self.padding_character = 0

		self.train_files = "corpus/train/"
		self.test_files = "corpus/test/"
		self.valid_files = "corpus/validation/"

		self.samples_number = {
			'full_train': 380968863,
			'full_test': 131424533,
			'full_valid': 131861863,
			'par_train': 84321880,
			'par_test': 29407143,
			'par_valid': 28882058,
		}

		self.map_no_diac = {
			'ă': 'a',
			'â': 'a',
			'Â': 'A',
			'Ă': 'A',
			'ț': 't',
			'Ț': 'T',
			'ș': 's',
			'Ș': 'S',
			'î': 'i',
			'Î': 'I',
			"ş": "ș",
			"Ş": "Ș",
			"ţ": "ț",
			"Ţ": "Ț",
		}
		self.all_in = {
			'ă','â','a',
			'Ă','Â','A',	
			'ț',
			'Ț',
			'ș',
			'Ș',
			'î',
			'Î',
			'i', 
			't',
			's',
			"ş",
			"Ş",
			"ţ",
			"Ţ",
			'S',
			'T',
			'I'
		}

		self.map_correct_diac = {
			"ş": "ș",
			"Ş": "Ș",
			"ţ": "ț",
			"Ţ": "Ț",
		}
		self.map_char_to_possible_chars = {
			'a': ['ă', 'â', 'a'], 
			'i': ['î', 'i'],
			's': ['ș', 's'], 
			't': ['ț', 't']
		}
		self.map_substitute_chars = {'"': '`'}
		self.characters_in_interest = {'a', 'i', 's', 't'}
		self.to_lower = {}
		self.rom_spacy = SpacyParser.get_instance().get_model(Lang.RO)
		
		self.create_lower_mapping()
		self.parse_args()

	# get case of the highest probability (returned by softmax)
	def get_case(self, p):
		case = 0
		for i in range(len(p)):
			if p[i] > p[case]:
				case = i
		return case

	def create_lower_mapping(self):
		for c in string.ascii_uppercase:
			self.to_lower[c] = c.lower()
		self.to_lower['Ș'] = 'ș'
		self.to_lower['Â'] = 'â'
		self.to_lower['Ă'] = 'ă'
		self.to_lower['Ț'] = 'ț'
		self.to_lower['Î'] = 'î'

	# case:
	#	0 -> ă
	#	1 -> â, î
	#	2 -> unmodified
	# 	3 -> ș, ț
	def get_label(self, i, clean_text_utf, original_text_utf):
		case = 2
		if clean_text_utf[i] == 'a':
			if original_text_utf[i] == 'ă':
				case = 0
			elif original_text_utf[i] == 'â':
				case = 1
			elif original_text_utf[i] == 'a':
				case = 2
		elif clean_text_utf[i] == 'i':
			if original_text_utf[i] == 'î':
				case = 1
			elif original_text_utf[i] == 'i':
				case = 2
		elif clean_text_utf[i] == 't':
			if original_text_utf[i] == 'ț':
				case = 3
			elif original_text_utf[i] == 't':
				case = 2
		elif clean_text_utf[i] == 's':
			if original_text_utf[i] == 'ș':
				if self.args.nr_classes == 5:
					case = 4
				else:
					case = 3
			elif original_text_utf[i] == 's':
				case = 2

		label = np.float32([0] * self.args.nr_classes)
		label[case] = np.float32(1.0)
		return label

	def bkt_all_words(self, index: int, clean_word: str, current_word: List) -> List:

		if index == len(clean_word):
			word = "".join(current_word)
			if self.args.use_dummy_word_embeddings == True:
				return [word]
			else:
				if word in self.model_embeddings.wv.vocab:
					return [word]
				else:
					return []
		else:
			L = []
			c = clean_word[index]
			if c in self.map_char_to_possible_chars:
				for ch in self.map_char_to_possible_chars[c]:
					current_word[index] = ch
					L += self.bkt_all_words(index + 1, clean_word, current_word)
			else:
				current_word[index] = c
				L += self.bkt_all_words(index + 1, clean_word, current_word)
			return L

	def get_avg_possible_word(self, clean_word):

		with self.dict_lock:
			if clean_word in self.dict_avg_words:
				return self.dict_avg_words[clean_word]
			
		count_diacritics_chars = 0
		for c in clean_word:
			if c in self.map_char_to_possible_chars:
				count_diacritics_chars += 1
		
		if count_diacritics_chars > self.limit_backtracking_characters:
			if self.args.use_dummy_word_embeddings == False:
				return np.float32(self.model_embeddings.wv[clean_word])
			else:
				return np.float32([0] * self.word_embedding_size)

		all_words = self.bkt_all_words(0, clean_word, ['a'] * len(clean_word))
		
		if len(all_words) > 0:
			if self.args.use_dummy_word_embeddings == False:
				return np.mean([np.float32(self.model_embeddings.wv[word]) for word in all_words], axis=0)
			else:
				return np.float32([0] * self.word_embedding_size)
		else:
			try:
				if self.args.use_dummy_word_embeddings == False:
					return np.float32(self.model_embeddings.wv[clean_word])
				else:
					return np.float32([0] * self.word_embedding_size)
			except:
				return np.float32([0] * self.word_embedding_size)

	def get_embeddings_sentence(self, clean_tokens_sentence, index_token):
		embeddings_sentence = []

		for i in range(index_token - self.window_sentence, index_token + self.window_sentence + 1):
			if i >= 0 and i < len(clean_tokens_sentence):
				token = clean_tokens_sentence[i]
				token_embedding = self.get_avg_possible_word(token)
				embeddings_sentence.append(token_embedding)
				with self.dict_lock:
					self.dict_avg_words[token] = token_embedding
			else:
				embeddings_sentence.append(np.float32([0] * self.word_embedding_size))
		return np.array(embeddings_sentence)

	# return an tuple input (window_char, embedding_token, embedding_sentence)
	def get_input_example(self, clean_text_utf, index_text, clean_tokens, index_sent, \
					index_last_sent, index_token):
		# window with characters
		w = []
		for j in range(index_text - self.window_character, index_text + self.window_character + 1):
			if j < 0 or j >= len(clean_text_utf):
				v1 = self.padding_character
			elif ord(clean_text_utf[j]) > self.max_unicode_allowed:
				v1 = self.replace_character
			else:
				v1 = ord(clean_text_utf[j])
			w.append(v1)
		# token 
		token = clean_tokens[index_sent][index_token]
		token_embedding = self.get_avg_possible_word(token)
		with self.dict_lock:
			self.dict_avg_words[token] = token_embedding

		# sentence 
		# if is the same sentence don't recompute it
		if index_last_sent is None or index_sent != index_last_sent:
			sentence_embedding = self.get_embeddings_sentence(clean_tokens[index_sent], index_token)
		else:
			sentence_embedding = None
		return (np.int32(w), token_embedding, sentence_embedding)

	def replace_char(self, c):

		if c in self.map_correct_diac:
			c = self.map_correct_diac[c]

		if c in self.to_lower:
			c = self.to_lower[c]

		if c in self.map_no_diac:
			c = self.map_no_diac[c]

		if ord(c) > 255:
			return chr(self.replace_character)
		elif c in self.map_substitute_chars:
			return self.map_substitute_chars[c]
		else:
			return c

	def replace_char_original(self, c):
		if c in self.map_correct_diac:
			c = self.map_correct_diac[c]

		if c in self.to_lower:
			c = self.to_lower[c]
		return c

	def count_chars_in_interest(self, s):
		cnt_chars = 0
		chars_in_int = []
		for c in s:
			character_c = chr(c)
			if character_c in self.characters_in_interest:
				cnt_chars += 1
				chars_in_int.append(character_c)
		return cnt_chars, chars_in_int

	def get_word_around_index(self, original_text_utf, index_text):
		s = []
		i = index_text
		while i >= 0 and original_text_utf[i].isalpha():
			i -= 1
		i += 1
		while i < len(original_text_utf) and original_text_utf[i].isalpha():
			s.append(original_text_utf[i])
			i += 1
		return "".join(s)

	def create_examples(self, original_text, is_test_dataset):
		drop_example = False

		try:
			original_text_utf = original_text.decode('utf-8')
			first_clean_text_utf = "".join([self.replace_char_original(c) for c in original_text_utf])
			# replace some strange characters which are modified by tokenization
			clean_text_utf = "".join([self.replace_char(c) for c in first_clean_text_utf])
			clean_sentences = nltk.sent_tokenize(clean_text_utf)
			clean_tokens = []
			clean_tokens_tag = []
			clean_tokens_dep = []

			# construct tokens
			for i in range(len(clean_sentences)):
				tokens = self.rom_spacy(clean_sentences[i])

				clean_tokens_sent = [token.text for token in tokens]
				tag_tokens_sent = [int(token.tag) for token in tokens]
				dep_tokens_sent = [int(token.dep) for token in tokens]
				
				#clean_tokens_sent = nltk.word_tokenize(clean_sentences[i])
				clean_tokens.append(clean_tokens_sent)
				clean_tokens_tag.append(tag_tokens_sent)
				clean_tokens_dep.append(dep_tokens_sent)

			index_text = 0 # current position in text
			index_sent = 0 # current sentence
			index_token = 0 # current token
			index_last_sent = None # last sentence computed

			# input and output lists
			clean_words = []
			original_words = []
			self.window_characters = []
			word_embeddings = []
			sentence_embeddings = []
			tag_tokens = []
			dep_tokens = []
			labels = []

			while index_sent < len(clean_tokens):
				clean_token = clean_tokens[index_sent][index_token]
				tag_token = clean_tokens_tag[index_sent][index_token]
				dep_token = clean_tokens_dep[index_sent][index_token]

				original_word = self.get_word_around_index(original_text_utf, index_text)
				i = 0
				# construct all inputs from the current token (letter(a, i, t, s) -> input)
				while i < len(clean_token):

					if clean_text_utf[index_text] in self.characters_in_interest:

						label = self.get_label(index_text, clean_text_utf, original_text_utf)
						#print(original_text_utf[index_text], label)
						win_char, word_emb, sent_emb = self.get_input_example(clean_text_utf, \
								index_text, clean_tokens, index_sent, index_last_sent, index_token)
						index_last_sent = index_sent
						if is_test_dataset == True:
							clean_words.append(clean_token)
							original_words.append(original_word)

						self.window_characters.append(win_char)
						word_embeddings.append(word_emb)
						tag_tokens.append(np.float32([tag_token]))
						dep_tokens.append(np.float32([dep_token]))
						# sentence already computed
						if sent_emb is None:
							sentence_embeddings.append(sentence_embeddings[-1])
						else:
							sentence_embeddings.append(sent_emb)					
						labels.append(label)
						#print(clean_text_utf[index_text], original_text_utf[index_text], label)

					if clean_text_utf[index_text] == clean_token[i]:
						index_text += 1
						i += 1
					else: # discard char in text
						index_text += 1

				if index_token == len(clean_tokens[index_sent]) - 1:
					index_token = 0
					index_sent += 1
				else:
					index_token += 1
		except Exception as e:
			drop_example = True
				
		# dummy values for empty sentence
		if len(self.window_characters) == 0 or drop_example == True:
			clean_words = ['a']
			self.window_characters = [np.int32([0] * (self.window_character * 2 + 1))]
			word_embeddings = [np.float32([0] * self.word_embedding_size)]
			sentence_embeddings = [np.array(\
				[np.float32([0] * self.word_embedding_size)] * (self.window_sentence * 2 + 1))]
			tag_tokens = [np.float32([0])]
			dep_tokens = [np.float32([0])]
			lab = np.float32([0] * self.args.nr_classes)
			lab[2] = np.float32(1.0)
			labels = [lab]
		if is_test_dataset == False:
			return (self.window_characters, word_embeddings, sentence_embeddings, tag_tokens, dep_tokens, labels)
		else:
			return (clean_words, self.window_characters, word_embeddings, sentence_embeddings, tag_tokens, dep_tokens, labels, original_words)

	def filter_null_strings(self, s):
		if len(s) == 0:
			return np.array([False])
		return np.array([True])

	def flat_map_f(self, a, b):
		return tf.data.Dataset.from_tensor_slices((a, b))

	def get_dataset(self, dpath, sess, is_test_dataset=False, restore=False, batch_1=False):

		if restore == False:
			input_files = tf.gfile.ListDirectory(dpath)
			for i in range(len(input_files)):			
				if self.args.corpus_rowiki == False and input_files[i].count('rowiki') > 0:
					input_files.remove(input_files[i])
					break
			
			for i in range(len(input_files)):
				input_files[i] = dpath + input_files[i]
		else:
			input_files = [dpath]

		dataset = tf.data.TextLineDataset(input_files)
		dataset = dataset.filter(lambda x:
		(tf.py_func(self.filter_null_strings, [x], tf.bool, stateful=False))[0])

		if is_test_dataset == True:
			datatype_returned = (tf.string, tf.int32, tf.float32, tf.float32, tf.float32, tf.float32,\
				tf.float32, tf.string)
		else:
			datatype_returned = (tf.int32, tf.float32, tf.float32, tf.float32, tf.float32,\
				tf.float32)

		dataset = dataset.map(lambda x:\
			tf.py_func(self.create_examples,\
						(x, is_test_dataset,),\
						datatype_returned,\
						stateful=False),\
			num_parallel_calls=self.CPUS)
		# map input - output tuple
		if is_test_dataset == True:
			dataset = dataset.map(lambda x1, x2, x3, x4, x5, x6, y1, y2:\
									((x1, x2, x3, x4, x5, x6), (y1, y2)),\
									num_parallel_calls=self.CPUS)
		else:
			dataset = dataset.map(lambda x1, x2, x3, x4, x5, y:\
									((x1, x2, x3, x4, x5), y),\
									num_parallel_calls=self.CPUS)
		dataset = dataset.flat_map(self.flat_map_f)
		
		# do not shuffle or batch test dataset
		if is_test_dataset == True:
			if batch_1 == False:
				dataset = dataset.batch(self.args.batch_size_restore)
			else:
				dataset = dataset.batch(1)
		else:
			dataset = dataset.shuffle(self.args.buffer_size_shuffle)
			dataset = dataset.batch(self.batch_size)
			dataset = dataset.prefetch(self.size_prefetch_buffer)

		return dataset

	def get_charr(self, simple_c, case):

		# 0 : ă
		# 1 : â, î
		# 2 : a, i, t, s
		# 3 : ț, ș
		if simple_c == 'a':
			if case == 0: # ă
				return 'ă'
			elif case == 1: # 
				return 'â'
			elif case == 2:
				return 'a'
		elif simple_c == 'i':
			if case == 1:
				return 'î'
			elif case == 2:
				return 'i'
		elif simple_c == 't':
			if case == 3:
				return 'ț'
			elif case == 2:
				return 't'
		elif simple_c == 's':
			if case == 3 and self.args.nr_classes == 4:
				return 'ș'
			elif case == 4 and self.args.nr_classes == 5:
				return 'ș'
			elif case == 2:
				return 's'

		logger.warning('Diacritics model may not be run with the proepr number of classes.')
		return 'a'

	def compute_prediction(self, correct_case, predicted_case, simple_c, precision_chars, recall_chars):
		correct_char = self.get_charr(simple_c, correct_case)
		pred_char = self.get_charr(simple_c, predicted_case)

		if pred_char == correct_char:
			# correct results
			recall_chars[correct_char][0] += 1
			precision_chars[correct_char][0] += 1
		precision_chars[pred_char][1] += 1
		recall_chars[correct_char][1] += 1

	def get_next_possible(self, all_txt, index_full_text):
		while True:
			index_full_text += 1
			if all_txt[index_full_text] in self.all_in:
				return index_full_text

	# restore char if you have the word
	def restore_char(self, pred_char, original_char, original_word):
		
		# don't modify char if already there (has diacritics)
		if original_char in self.map_no_diac:
			return original_char
		# don't modify the word if is only uppercase letters
		elif original_word.isupper():
			return original_char
		elif original_char.isupper():
			return pred_char.upper()
		return pred_char

	# restore char without word
	def restore_char_simple(self, pred_char, original_char):
		# don't modify char if already there
		if original_char in self.map_no_diac:
			return original_char
		elif original_char.isupper():
			return pred_char.upper()
		return pred_char

	# restoration expects a file
	def restore_file(self, sess, model, file_path, nr_batches, batch_1=False):

		# print('restoring file {}'.format(file_path))

		all_txt = []
		with open(file_path, 'r', encoding='utf-8') as f:
			for line in f:
				all_txt.append(list(line))
		all_txt = list(itertools.chain(*all_txt))

		if batch_1 == True:
			self.batch_sizee = 1
		else:
			self.batch_sizee = self.args.batch_size_restore

		dt_test = self.get_dataset(file_path, sess, True, True, batch_1=batch_1)
		iterator_test = dt_test.make_initializable_iterator()
		
		sess.run(iterator_test.initializer)
		test_inp_pred, _ = iterator_test.get_next()
		_, test_char_window_pred, test_words_pred, test_sentence_pred, _, _ = test_inp_pred
		test_char_window_pred = tf.reshape(test_char_window_pred, [self.batch_sizee, self.window_character * 2 + 1])
		test_words_pred = tf.reshape(test_words_pred, [self.batch_sizee, self.word_embedding_size])
		test_sentence_pred = tf.reshape(test_sentence_pred, [self.batch_sizee, self.window_sentence * 2 + 1, self.word_embedding_size])

		input_list = self.get_input_list(test_char_window_pred, test_words_pred, test_sentence_pred, None, None)
		predictions = model.predict(x=input_list,
					verbose=1,
					steps=nr_batches)
		sess.run(iterator_test.initializer)
		index_full_text = -1

		for current_prediction in range(self.batch_sizee * nr_batches):
			pred_vector = predictions[current_prediction]
			predicted_case = self.get_case(pred_vector)
			index_full_text = self.get_next_possible(all_txt, index_full_text)
			c = self.replace_char_original(all_txt[index_full_text])
			c = self.replace_char(c)
			pred_char = self.get_charr(c, predicted_case)
			all_txt[index_full_text] = self.restore_char_simple(pred_char, all_txt[index_full_text])
		res = "".join(all_txt)
		return res

	# restoration is done in batches
	# the text is splitted into 2 files - one for the bog batches and with batches == 1
	def restore_diacritics(self, sess, model):

		logger.info('Starting restoring diacritics')
		path1 = "".join([random.choice(string.ascii_uppercase + string.digits) for _ in range(10)]) +  '_big_batches.txt'
		path2 = "".join([random.choice(string.ascii_uppercase + string.digits) for _ in range(10)]) + '_small_batches.txt'
		out_file = self.args.output_file_restore

		txt_file = self.args.restore_diacritics
		all_txt = []
		
		with open(txt_file, 'r', encoding='utf-8') as f:
			for line in f:
				all_txt.append(list(line))
		# print('input text read: {} lines'.format(len(all_txt)))
		all_txt = list(itertools.chain(*all_txt))
		all_txt_chars = 0

		# count nr of predictions (all possible)
		for c in all_txt:
			if c in self.all_in:
				all_txt_chars += 1
		nr_batches = all_txt_chars // self.args.batch_size_restore
		partial_txt_chars = 0
		# print('nr batches: {}'.format(nr_batches))

		if all_txt_chars >= self.args.batch_size_restore:

			for i, c in enumerate(all_txt):
				if c in self.all_in:
					partial_txt_chars += 1
					if partial_txt_chars == nr_batches * self.args.batch_size_restore:
						txt_big_batches = all_txt[:i+1]
						txt_small_batches = all_txt[i+1:]
						break
			with open(path1, 'w', encoding='utf-8') as f:
				f.write("".join(txt_big_batches))
				
			with open(path2, 'w', encoding='utf-8') as f:
				f.write("".join(txt_small_batches))
			
			s1 = self.restore_file(sess, model, path1, nr_batches, batch_1=False)
			s2 = self.restore_file(sess, model, path2, all_txt_chars - partial_txt_chars, batch_1=True)
			os.remove(path1)
			os.remove(path2)

			with open(out_file, 'w', encoding='utf-8') as f:
				f.write(s1 + s2)
		else:
			s = self.restore_file(sess, model, txt_file, all_txt_chars, batch_1=True)
			with open(out_file, 'w', encoding='utf-8') as f:
				f.write(s)
		logger.info('Successfully restored diacritics')

	def compute_test_accuracy(self, sess, model):
		dt_test = self.get_dataset(self.test_files, sess, True)
		iterator_test = dt_test.make_initializable_iterator()

		sess.run(iterator_test.initializer)
		nr_test_batches = self.args.number_samples_test

		test_inp_pred, _ = iterator_test.get_next()
		test_string_word_pred, test_char_window_pred, test_words_pred, test_sentence_pred, _, _ = test_inp_pred
		predictions = model.predict(x=[test_char_window_pred, test_words_pred, test_sentence_pred],
					verbose=1,
					steps=nr_test_batches)

		current_test_batch = 0
		sess.run(iterator_test.initializer)
		test_next_element = iterator_test.get_next()
		prediction_index = 0
		total_words = 0
		correct_predicted_words = 0
		correct_predicted_chars = 0
		wrong_predicatd_chars = 0
		precision_chars = {'ă': [0, 0], 'â': [0, 0], 'a': [0, 0], 'i': [0, 0], 'î': [0, 0], 's': [0, 0],\
							'ș': [0, 0], 't': [0, 0], 'ț': [0, 0]}
		recall_chars = {'ă': [0, 0], 'â': [0, 0], 'a': [0, 0], 'i': [0, 0], 'î': [0, 0], 's': [0, 0],\
							'ș': [0, 0], 't': [0, 0], 'ț': [0, 0]}

		wrong_restoration_words  = {}
		correct_restoration_words = {}
		acc_restoration_word = {}
		all_words= set()

		while True:
			try:
				test_inp, test_out = sess.run(test_next_element)
				test_string_word, _, _, _ = test_inp
				current_test_batch += 1
				index_batch = 0
				if current_test_batch % 100 == 0:
					print('batch {} out of {}'.format(current_test_batch, nr_test_batches))
				while index_batch < len(test_string_word):
					# skip last word no matter what
					word = test_string_word[index_batch]
					all_words.add(word)
					nr_chars_in_word, chars_in_int = self.count_chars_in_interest(word)
					if nr_chars_in_word > len(test_string_word) - index_batch:
						prediction_index += len(test_string_word) - index_batch
						break

					correct_prediction_word = True
					for i in range(nr_chars_in_word):
						pred_vector = predictions[prediction_index]
						predicted_case = self.get_case(pred_vector)
						correct_case = self.get_case(test_out[index_batch][0])
						index_batch += 1
						prediction_index += 1

						if predicted_case != correct_case:
							correct_prediction_word = False
							wrong_predicatd_chars += 1
						else:
							correct_predicted_chars += 1
						self.compute_prediction(correct_case, predicted_case, chars_in_int[i], precision_chars, recall_chars)

					total_words += 1

					if correct_prediction_word == True:
						correct_predicted_words += 1
						if word in correct_restoration_words:
							correct_restoration_words[word] += 1
						else:
							correct_restoration_words[word] = 1
					else:
						if word in wrong_restoration_words:
							wrong_restoration_words[word] += 1
						else:
							wrong_restoration_words[word] = 1
						
				if current_test_batch == nr_test_batches:
					break
			except tf.errors.OutOfRangeError:
				break

		index_word = 0
		print('highest missed words: ')
		for key, value in sorted(wrong_restoration_words.items(), key=lambda x: x[1], reverse=True):
			correct = 0
			if key in correct_restoration_words:
				correct = correct_restoration_words[key]
			print("word '"  + key.decode('utf-8') + "' wrong: " +  str(value) + \
				' correct: ' + str(correct) + ' accuracy: ' + str(1.0 * correct / (value + correct)))
			index_word += 1
			if index_word == self.args.top_wrong_words_restoration:
				break

		print('precision per character: ')
		
		for key, values in precision_chars.items():
			if values[1] != 0:
				p = values[0] / values[1]
				print(key + ': ' + str(p))
				precision_chars[key] = p
		

		print('recall per character: ')

		for key, values in recall_chars.items():
			if values[1] != 0:
				r = values[0] / values[1]
				print(key + ': ' + str(r))
				recall_chars[key] = r
		
		print('F1 measure per character: ')
		for key, r in recall_chars.items():
				p = precision_chars[key]
				if p != 0 and r != 0:
					f1 = 2 * p * r / (p + r)
					print(key + ': ' + str(f1))

		(char_acc, word_acc) = (correct_predicted_chars / (correct_predicted_chars + wrong_predicatd_chars), correct_predicted_words / total_words)
		print("char acc: " + str(char_acc) + ", word accuracy: " + str(word_acc) + ' ')
		return char_acc, word_acc

	def set_up_folders_saved_models(self):
		full_path_dir = self.args.folder_saved_model_per_epoch
		if self.args.save_model == True:
			if os.path.exists(full_path_dir) == False:
				os.makedirs(full_path_dir)
			elif self.args.load_model_name is None:
				print('a folder with the same name (' + self.args.folder_saved_model_per_epoch +\
					') for saving model already exists, delete it to continue or give other name to the folder of the saved model')
				exit(0)

	def parse_args(self):
		# when specify -load, specify also the nr of classes and if the model uses chars, words, sent
		# to restore diacritics run: 
		# python model_diacritice.py -buff 1000 -no_fast -load saved_models_diacritice/chars16-32-4classes -no_word -no_sent -classes 4 -no_dep -no_tag -restore raw_text.txt
		
		parser = argparse.ArgumentParser(description='Run diacritics model')
		parser.add_argument('-s', dest="save_model", action='store_false', default=True,\
							help="save the model (and weights), default=true")
		parser.add_argument('-f', dest="folder_saved_model_per_epoch",\
							action='store', default="char_word_sentence",\
							help="name of the folder to store the weights, default: char_word_sentence")
		parser.add_argument('-c', dest="corpus_rowiki",\
							action='store_true', default=False,\
							help="if you want to use rowiki corpus, beside parliament corpus, default=false")
		parser.add_argument('-test', dest="do_test",\
							action='store_true', default=False,\
							help="if you want to run test dataset, default=false")
		parser.add_argument('-n_test', dest="number_samples_test",\
							action='store', default=self.samples_number['par_test'] // self.batch_size, type=int,\
							help="number of samples for test accuracy, if -test is not set \
							this does not have any effect, default=100000")
		parser.add_argument('-e', dest="epochs",\
							action='store', default=20, type=int,\
							help="number of epochs, default=20")
		parser.add_argument('-r', dest="reset_iterators_every_epochs",\
							action='store', default=10, type=int,\
							help="reset the iterators for the dataset every nr epochs, default=10")
		parser.add_argument('-buff', dest="buffer_size_shuffle",\
							action='store', default=100000, type=int,\
							help="size of the buffer for shuffle, default=100000")
		parser.add_argument('-no_fast', dest="use_dummy_word_embeddings",\
							action='store_true', default=False,\
							help="use dummy word embeddings instead of fasttext, default=false")
		parser.add_argument('-load', dest="load_model_name",\
							action='store', default=None,\
							help="load presaved model and weights\
							, specify just the folder name, it will take the last epoch file,\
							default=None")
		parser.add_argument('-no_train_valid', dest="run_train_validation",\
							action='store_false', default=True,\
							help="run train and validation, if false you should set -load model param\
							, default=True")
		parser.add_argument('-mgpu', dest="percent_memory_gpu",\
							action='store', default=0.2, type=float,\
							help="percentage of the gpu memory to use, default=0.2")
		parser.add_argument('-wrong', dest="top_wrong_words_restoration",\
							action='store', default=30, type=int,\
							help="hardest words to restore, default=30")
		parser.add_argument('-no_char', dest="use_window_characters",\
							action='store_false', default=True,\
							help="if model should use window of characters, default=True")
		parser.add_argument('-no_word', dest="use_word_embedding",\
							action='store_false', default=True,\
							help="if model should use word embeddings, default=True")
		parser.add_argument('-no_sent', dest="use_sentence_embedding",\
							action='store_false', default=True,\
							help="if model should use sentence embedding, default=True")
		parser.add_argument('-no_tags', dest="use_tags",\
							action='store_false', default=True,\
							help="if model should use tags of the words, default=True")
		parser.add_argument('-no_deps', dest="use_deps",\
							action='store_false', default=True,\
							help="if model should use dependencies of the words, default=True")
		parser.add_argument('-hidden', dest="hidden_neurons",\
							action='append', type=int,\
							help="number of neurons on the hidden layer, no default")
		parser.add_argument('-classes', dest="nr_classes", default=4,\
							action='store', type=int,\
							help="number of classes to be used (4 or 5), default=4")
		parser.add_argument('-restore', dest="restore_diacritics", 
							action='store', default=None,\
							help="name of the file to restore diacritics")
		parser.add_argument('-batch_size_restore', dest="batch_size_restore", 
							action='store', type=int, default=64,\
							help="batch size ised for restoration")
		parser.add_argument('-output_file_restore', dest="output_file_restore", 
							action='store',  default='tmp_res.txt',\
							help="output of the file with diacritics")
		self.args = parser.parse_args()
		self.args.folder_saved_model_per_epoch += '/'

		
		if self.args.restore_diacritics is not None:
			# by default use the best model
			
			self.args.save_model = False
			self.args.do_test = False
			self.args.buffer_size_shuffle = 100
			self.args.run_train_validation = False

		if self.args.nr_classes != 4 and self.args.nr_classes != 5:
			print('classes has to be either 4 or 5, exit')
			exit(0)

		# for k in self.args.__dict__:
		# 	if self.args.__dict__[k] is not None:
		# 		print(k, '->', self.args.__dict__[k])

	def get_number_samples(self):
		if self.args.corpus_rowiki == False:
			inp_batches_train = self.samples_number['par_train'] // self.batch_size
			inp_batches_test = self.samples_number['par_test'] // self.batch_size
			inp_batches_valid = self.samples_number['par_valid'] // self.batch_size
		else:
			inp_batches_train = self.samples_number['full_train'] // self.batch_size
			inp_batches_test = self.samples_number['full_test'] // self.batch_size
			inp_batches_valid = self.samples_number['full_valid'] // self.batch_size
		return inp_batches_train, inp_batches_test, inp_batches_valid

	def get_input_list(self, characters_bi_lstm_layer, word_embeddings_layer, sentence_bi_lstm_layer, tags, deps):
		
		input_list = []

		if self.args.use_window_characters == True:
			input_list.append(characters_bi_lstm_layer)

		if self.args.use_word_embedding == True:
			input_list.append(word_embeddings_layer)

		if self.args.use_sentence_embedding == True:
			input_list.append(sentence_bi_lstm_layer)
		
		if self.args.use_tags == True:
			input_list.append(tags)

		if self.args.use_deps == True:
			input_list.append(deps)

		if len(input_list) == 1:
			return input_list[0]

		return input_list

	# class AttentionLayer(keras.layers.Layer):

	# 	def __init__(self, **kwargs):
	# 		super(AttentionLayer, self).__init__(**kwargs)

	# 	# multiple inputs represented as a list (of tuples?)
	# 	def build(self, input_shape):
	# 		assert isinstance(input_shape, list)
	# 		# Create a trainable weight variable for this layer.
	# 		# 
	# 		self.kernel = self.add_weight(name='kernel',
	# 									shape=(self.character_embedding_size, self.characters_cell_size * 2),
	# 									initializer='uniform',
	# 									trainable=True)
	# 		super(AttentionLayer, self).build(input_shape)  # Be sure to call this at the end

	# 	def call(self, x):
	# 		assert isinstance(x, list)
	# 		# W = windows size
	# 		# C = cell size
	# 		# E = embedding size
	# 		# inputt has shape (self.batch_size, W, E) 
	# 		# hidden has shape (self.batch_size, W, C * 2)
	# 		# kernel has shape (E, C * 2)
	# 		inputt, hidden = x
	# 		# inputt * kernel * hidden_states.T
	# 		return K.batch_dot(K.batch_dot(inputt, K.repeat_elements(K.expand_dims(self.kernel, axis=0), self.batch_size, axis=0)),\
	# 				K.permute_dimensions(hidden, (0, 2, 1)))
	# 		# res i j = correlation between input i and hidden state j

	# 	def compute_output_shape(self, input_shape):
	# 		assert isinstance(input_shape, list)
	# 		return (self.batch_size, self.window_character * 2 + 1, self.window_character * 2 + 1)

	# construct the model 
	def construct_model(self, sess):
		
		last_epoch = 0
		if self.args.load_model_name is not None:
			
			# folder_path_with_epochs = self.args.load_model_name 
			# epochs_files = os.listdir(folder_path_with_epochs)
			# sorted_epochs_files = sorted(epochs_files)
			# self.args.load_model_name = os.path.join(self.args.load_model_name, sorted_epochs_files[-1])
			# print('load model name: {}'.format(self.args.load_model_name))
			model = keras.models.load_model(self.args.load_model_name)
			#last_epoch = len(sorted_epochs_files)
		else:
			vocabulary_size = self.max_unicode_allowed + 1
			# character window 
			input_character_window = keras.layers.Input(shape=(self.window_character * 2 + 1,))
			character_embeddings_layer = keras.layers.Embedding(\
										input_dim=vocabulary_size,\
										output_dim=self.character_embedding_size)(input_character_window)

			character_lstm_layer = keras.layers.LSTM(
									units=self.characters_cell_size,\
									input_shape=(self.window_character * 2 + 1, self.character_embedding_size,),
									return_sequences=False)
			
			characters_bi_lstm_layer = keras.layers.Bidirectional(
									layer=character_lstm_layer,\
									merge_mode="concat")(character_embeddings_layer)
			# word token					
			word_embeddings_layer = keras.layers.Input(shape=(self.word_embedding_size,))

			# sentence token
			sentence_embeddings_layer = keras.layers.Input(shape=((self.window_sentence * 2 + 1, self.word_embedding_size,)))
			sentence_lstm_layer = keras.layers.LSTM(units=self.sentence_cell_size,\
													input_shape=(self.window_sentence * 2 + 1, self.word_embedding_size,))	

			sentence_bi_lstm_layer = keras.layers.Bidirectional(layer=sentence_lstm_layer,\
																merge_mode="concat")(sentence_embeddings_layer)
			# tags 
			tags = keras.layers.Input(shape=(1,),\
									dtype='float32')
			# deps
			deps = keras.layers.Input(shape=(1,),\
									dtype='float32')
			# merged
			input_list = self.get_input_list(characters_bi_lstm_layer,\
				word_embeddings_layer, sentence_bi_lstm_layer, tags, deps)
			if len(input_list) > 1:
				merged_layer = keras.layers.concatenate(input_list, axis=-1)
			else:
				merged_layer = input_list[0]
			prev_layer = merged_layer
			#prev_layer = keras.layers.Flatten()(prev_layer)
			#attention = AttentionLayer()([character_embeddings_layer, prev_layer])
			#attention = keras.layers.Flatten()(attention)
			attention = prev_layer

			# hidden layers
			for h_neurons in self.args.hidden_neurons:
				attention = keras.layers.Dense(h_neurons, activation='tanh')(attention)
			output = keras.layers.Dense(self.args.nr_classes, activation='softmax')(attention)

			# [input_character_window, word_embeddings_layer, sentence_embeddings_layer]
			model_input = self.get_input_list(input_character_window, word_embeddings_layer, sentence_embeddings_layer,\
				tags, deps)

			model = keras.models.Model(inputs=model_input,\
									outputs=output)
			model.compile(optimizer='adam',\
						loss='categorical_crossentropy',\
						metrics=['accuracy'])
			print(model.summary())
		return model, last_epoch

	def run_task(self):
		inp_batches_train, inp_batches_test, inp_batches_valid = self.get_number_samples()
		self.set_up_folders_saved_models()

		if self.args.use_dummy_word_embeddings == False:
			self.model_embeddings = FastTextWrapper.load_fasttext_format(self.fast_text)

		config = tf.ConfigProto()
		config.gpu_options.per_process_gpu_memory_fraction = self.args.percent_memory_gpu

		with tf.Session(config=config) as sess:

			if inp_batches_test < self.args.number_samples_test:
				print("cannot start, too many test samples given, has to be lower than "\
						+ str(inp_batches_test))

			print("char, word, sentence - char cell:{}, word cell: {}, hidden: {}"\
				.format(self.characters_cell_size, self.sentence_cell_size, self.neurons_dense_layer_after_merge))

			model, last_epoch = self.construct_model(sess)
			# run test and validation 
			if self.args.run_train_validation == True:

				dt_train = self.get_dataset(self.train_files, sess)
				dt_valid = self.get_dataset(self.valid_files, sess)
				iterator_train = dt_train.make_initializable_iterator()
				iterator_valid = dt_valid.make_initializable_iterator()

				for i in range(last_epoch, last_epoch + self.args.epochs):

					print('epoch: ' + str(i + 1))
					# reset iterators
					if (i - last_epoch) % self.args.reset_iterators_every_epochs == 0:
						print('resseting iterators')
						sess.run(iterator_valid.initializer)
						valid_inp, valid_out = iterator_valid.get_next()
						valid_char_window, valid_words, valid_sentence, valid_tags, valid_deps = valid_inp

						sess.run(iterator_train.initializer)
						train_inp, train_out = iterator_train.get_next()
						train_char_window, train_words, train_sentence, train_tags, train_deps = train_inp
					
					# train an epoch
					train_input = self.get_input_list(train_char_window, train_words, train_sentence,\
						train_tags, train_deps)
					model.fit(\
						train_input,\
						[train_out],\
						steps_per_epoch=inp_batches_train//self.args.reset_iterators_every_epochs,\
						epochs=1,\
						verbose=1)

					# save weights
					if self.args.save_model == True:
						print('saving model (and weights)')
						full_path_epoch_weights = os.path.join(self.args.folder_saved_model_per_epoch, 'epoch_' + str(i) + '.h5')
						model.save(full_path_epoch_weights)
					# validate 
					valid_input = self.get_input_list(valid_char_window, valid_words, valid_sentence,
						valid_tags, valid_deps)
					[valid_loss, valid_acc] = model.evaluate(valid_input,\
												valid_out,\
												verbose=1,\
												steps=inp_batches_valid//self.args.reset_iterators_every_epochs)
					print("validation - loss: " + str(valid_loss) +  " acc: " + str(valid_acc))
			# test
			if self.args.do_test:
				self.compute_test_accuracy(sess, model)
			if self.args.restore_diacritics:
				self.restore_diacritics(sess, model)

if __name__ == "__main__":
	diacritics = Diacritics()
	diacritics.args.use_window_characters = True
	diacritics.args.use_dummy_word_embeddings = True
	diacritics.args.use_sentence_embedding = False
	diacritics.args.use_word_embedding = False
	diacritics.args.buffer_size_shuffle = 1000 
	diacritics.args.load_model_name = './saved_models_diacritice/chars16-32-4classes/'
	diacritics.args.nr_classes = 4
	diacritics.args.use_tags = False
	diacritics.args.use_deps = False
	diacritics.args.restore_diacritics = 'book.txt'
	diacritics.args.batch_size_restore = 1024

	diacritics.args.save_model = False
	diacritics.args.do_test = False
	diacritics.args.run_train_validation = False
	diacritics.args.output_file_restore = 'tmp_res2.txt'

	diacritics.run_task()
