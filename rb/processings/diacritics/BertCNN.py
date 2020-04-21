#pylint: disable=import-error
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Concatenate, Convolution1D, GlobalMaxPooling1D, Embedding, Dropout, Lambda
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.callbacks import TensorBoard
import sys
import numpy as np
from tensorflow.keras import backend as K
import rb.processings.diacritics.utils as utils
import functools

class BertCNN(object):
	"""
	Class to implement simple Bert + Character Level Convolutional Neural Network
	The model is used to classify diacritics
    """


	def __init__(self, window_size, alphabet_size, embedding_size, conv_layers, num_of_classes, batch_max_sentences, batch_max_windows,
				 bert_trainable, cnn_dropout_rate, bert_wrapper, learning_rate, optimizer='adam', loss='categorical_crossentropy'):
		"""
        Initialization for the Bert + Character Level CNN model.
        Args:
			CNN side
            window_size (int): Size of window
            alphabet_size (int): Size of alphabets to create embeddings for
            embedding_size (int): Size of embeddings
            conv_layers (list[list[int]]): List of Convolution layers for model
            cnn_dropout_rate (float): Dropout Rate for CNN
            
			Bert side
			bert_wrapper (obj): Bert wrapper
			bert_trainable (bool): Whether to train BERT or not
			batch_max_sentences (int): Maximum sentences in batch
			batch_max_windows (int): Maximum windows in batch



			num_of_classes (int): Number of classes in data
			optimizer (str): Training optimizer
            loss (str): Loss function
			learning_rate (float): Learning rate to use for training

        """
		self.window_size = window_size
		self.alphabet_size = alphabet_size
		self.embedding_size = embedding_size
		self.conv_layers = conv_layers
		self.total_number_of_filters = functools.reduce(lambda x,y: x+y[0], conv_layers, 0)
		self.num_of_classes = num_of_classes
		self.cnn_dropout_rate = cnn_dropout_rate
		self.learning_rate = learning_rate
		
		self.bert_wrapper = bert_wrapper
		self.bert_wrapper.bert_layer.trainable = bert_trainable
		self.batch_max_sentences = batch_max_sentences
		self.batch_max_windows = batch_max_windows

		if optimizer == "adam":
			self.optimizer = keras.optimizers.Adam(lr=self.learning_rate)

		if loss == "categorical_crossentropy":
			self.loss = keras.losses.CategoricalCrossentropy(from_logits=False)

		self._build_model()  # builds self.model variable

	
	def _build_embedding_mask(self):
		embedding_mask_weights = np.zeros((self.alphabet_size, self.num_of_classes))
		# a -> a, ă, â
		embedding_mask_weights[2] = [1,1,1,0,0]
		# s -> s, ș
		embedding_mask_weights[10] = [1,0,0,1,0]
		# t -> t, ț
		embedding_mask_weights[13] = [1,0,0,0,1]
		# i -> i, î
		embedding_mask_weights[16] = [1,0,1,0,0]
		return embedding_mask_weights
	

	def _build_model(self):
		"""
        Build and compile the Bert + Character Level CNN model
        Returns: None
        """

		# Input layers
		input_bert_ids = Input(shape=(self.batch_max_sentences, self.bert_wrapper.max_seq_len), name='bert_input_ids')
		input_bert_seg = Input(shape=(self.batch_max_sentences, self.bert_wrapper.max_seq_len), name='bert_segment_ids')
		input_token_ids = Input(shape=(self.batch_max_windows,), name='token_ids', dtype='int32')
		input_sent_ids = Input(shape=(self.batch_max_windows,), name='sent_ids', dtype='int32')
		input_mask = Input(shape=(self.batch_max_windows,), name='mask', dtype='float32')
		input_char_windows = Input(shape=(self.batch_max_windows, self.window_size), name='char_windows')
		
		
		keras_internal_batch_size = K.shape(input_token_ids)[0]

		##########################################################################
		######################  Bert  ############################################
		input_bert_ids_reshaped = tf.reshape(input_bert_ids, shape=(-1, self.bert_wrapper.max_seq_len))
		input_bert_seg_reshaped = tf.reshape(input_bert_seg, shape=(-1, self.bert_wrapper.max_seq_len))
		# shape = (?batch_size x max_sent, max_seq_len)

		bert_output = self.bert_wrapper.bert_layer(input_bert_ids_reshaped, input_bert_seg_reshaped)
		# bert_output = (?batch_size x max_sent, bert_max_seq_len, bert_hidden_size)
		bert_output = tf.reshape(bert_output, shape=(-1, self.batch_max_sentences, self.bert_wrapper.max_seq_len, self.bert_wrapper.hidden_size), name="bert_output")
		# bert_output = (?batch_size, max_sent, bert_max_seq_len, bert_hidden_size)

		##########################################################################

		##########################################################################
		######################  CharCNN  #########################################
		embedding_mask_weights = self._build_embedding_mask()
		input_char_windows_reshaped = tf.reshape(input_char_windows, shape=(-1, self.window_size))
		# shape = (?batch_size x max_windows, window_size)
		# char mask
		char_mask = Embedding(self.alphabet_size, self.num_of_classes, input_length=1, trainable=False, weights=[embedding_mask_weights], name="mask_embedding")(input_char_windows_reshaped[:, (self.window_size-1)//2])				
		char_mask = tf.reshape(char_mask,(-1, self.batch_max_windows, self.num_of_classes))
		# Embedding layer
		x = Embedding(self.alphabet_size, self.embedding_size, input_length=self.window_size, trainable=True, name="sequence_embedding")(input_char_windows_reshaped)

		# x = (?batch_size, window_size, embedding_size)
		# Convolution layers
		convolution_output = []
		for num_filters, filter_width in self.conv_layers:
			conv = Conv1D(filters=num_filters, kernel_size=filter_width, activation='tanh',
									name='Conv1D_{}_{}'.format(num_filters, filter_width))(x)
			# conv = (?batch_size, window_size-filter_size+1, num_filters)
			pool = GlobalMaxPooling1D(name='MaxPoolingOverTime_{}_{}'.format(num_filters, filter_width))(conv)
			# pool = (?batch_size, num_filters)
			convolution_output.append(pool)
		x = Concatenate()(convolution_output)
		# x = (?batch_size, total_number_of_filters)
		char_cnn_output = Dropout(rate=self.cnn_dropout_rate)(x)
		char_cnn_output = tf.reshape(char_cnn_output, shape=(-1, self.batch_max_windows, self.total_number_of_filters), name="char_cnn_output")
		# char_cnn_otput = (?batch_size, max_windows, total_filters)
		##########################################################################
		
		
		# get bert tokens coresponding to sent_ids and token_ids
		batch_indexes = tf.range(0, keras_internal_batch_size)
		batch_indexes = tf.reshape(batch_indexes, (keras_internal_batch_size,1))
		batch_indexes = tf.tile(batch_indexes, (1,self.batch_max_windows))
		indices = tf.stack([batch_indexes, input_sent_ids, input_token_ids], axis = 2)
		bert_tokens = tf.gather_nd(bert_output, indices)
		# apply bert dropout here?
		# bert_tokens = (?batch_size, max_windows, bert_hidden_size)
		bert_cnn_concatenation = Concatenate()([bert_tokens, char_cnn_output])
		
		# Output layer
		predictions = Dense(self.num_of_classes, activation='softmax')(bert_cnn_concatenation)
		# mask predictions based on middle char 
		masked_predictions = keras.layers.multiply([predictions, char_mask])
		# mask prediction based on window mask
		extended_mask = tf.reshape(input_mask, (-1, self.batch_max_windows, 1))
		print(extended_mask)
		print()
		extended_mask = tf.tile(extended_mask, [1, 1, self.num_of_classes])

	
		masked_predictions = keras.layers.multiply([masked_predictions, extended_mask])
		# print(masked_predictions)
		# flatten_masked_predictions = tf.reshape(masked_predictions, shape=(-1, self.batch_max_windows, self.num_of_classes))
		flatten_masked_predictions = masked_predictions
		# flatten_masked_prediction = (?batch_size, max_windows, num_of_classes)
		# print(flatten_masked_predictions)
		# sys.exit()
		# Build and compile model
		model = Model(inputs=[input_bert_ids, input_bert_seg, input_token_ids, input_sent_ids, input_mask, input_char_windows], outputs=flatten_masked_predictions)

		model.compile(optimizer=self.optimizer, loss=self.loss, metrics=[tf.keras.metrics.categorical_accuracy])

		# self.bert_wrapper.load_weights()
		self.model = model
		print("Bert+CharCNN model built: ")
		self.model.summary()

		

	def train(self, train_dataset, train_batch_size, train_size, dev_dataset, dev_batch_size, dev_size, epochs, file_evalname, char_to_id_dict):

		best_wa_dia = -1
		best_wa_all = -1
		best_ca_dia = -1
		best_ca_all = -1
		best_epoch = -1

		train_batch_size = 1
		dev_batch_size = 1

		for i in range(epochs):
			print("EPOCH ", (i+1))
			self.model.fit(train_dataset, steps_per_epoch=train_size//train_batch_size, epochs=1, verbose=1)
			self.model.evaluate(dev_dataset, steps=dev_size//dev_batch_size, verbose=1)
			print("---------------")
			sys.exit()
			wa_dia, wa_all, ca_dia, ca_all, predicted_words = utils.evaluate_model_on_file(self.model, file_evalname, char_to_id_dict, self.window_size)
			if wa_dia > best_wa_dia:
				best_wa_dia = wa_dia
				best_wa_all = wa_all
				best_ca_dia = ca_dia
				best_ca_all = ca_all
				best_epoch = i+1
				self.model.save('rb/processings/diacritics/models/model_ws{0}_tbs{1}_embdim{2}_lr{3}_drop{4}.h5'.format(self.window_size, train_batch_size, self.embedding_size, self.learning_rate, self.cnn_dropout_rate))

				outfile_name = "rb/processings/diacritics/models/output_{5}_model_ws{0}_tbs{1}_embdim{2}_lr{3}_drop{4}.txt".format(self.window_size, train_batch_size, self.embedding_size, self.learning_rate, self.cnn_dropout_rate, file_evalname.split("/")[-1].split(".")[0])
				# also write to file
				with open(outfile_name , "w", encoding="utf-8") as outfile:
					for word in predicted_words:
						if word[-1] == "\n":
							outfile.write(word)
						else:
							outfile.write(word + " ")
			
			print("Best model: epoch =", best_epoch, "best word_accuracy_dia =", format(best_wa_dia, '.4f'), "best word_accuracy_all =", format(best_wa_all, '.4f'), 
							"best char_accuracy_dia =", format(best_ca_dia, '.4f'), "best char_accuracy_all =", format(best_ca_all, '.4f'))
			print("---------------")

class weighted_categorical_crossentropy(object):
	"""
	A weighted version of keras.objectives.categorical_crossentropy

	Variables:
		weights: numpy array of shape (C,) where C is the number of classes

	Usage:
		loss = weighted_categorical_crossentropy(weights).loss
		model.compile(loss=loss,optimizer='adam')
	"""

	def __init__(self,weights):
		self.weights = K.variable(weights)
        
	def loss(self, y_true, y_pred):
		y_true = K.print_tensor(y_true)
		y_pred = K.print_tensor(y_pred)

		# scale preds so that the class probas of each sample sum to 1
		y_pred = y_pred / K.sum(y_pred, axis=-1, keepdims=True)
		# y_pred = K.print_tensor(y_pred)

		# clip
		y_pred = K.clip(y_pred, K.epsilon(), 1)
		# y_pred = K.print_tensor(y_pred)
		
		# calc
		loss = y_true*K.log(y_pred)*self.weights
		# loss = K.print_tensor(loss)
		loss =-K.sum(loss,-1)
		# loss = K.print_tensor(loss)
		# sys.exit()
		return loss