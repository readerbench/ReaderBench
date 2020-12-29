#pylint: disable=import-error
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Concatenate, Convolution1D, GlobalMaxPooling1D, Embedding, Dropout, Lambda, Flatten, Add
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


	def __init__(self, window_size, alphabet_size, embedding_size, conv_layers, fc_hidden_size, num_of_classes, batch_max_sentences, batch_max_windows,
				 bert_trainable, cnn_dropout_rate, bert_wrapper, learning_rate, init_model, optimizer='adam', loss='categorical_crossentropy'):
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
			

			init_model (string): Name of model to start training from
			fc_hidden_size (int): Size of hidden layer between features and prediction

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
		self.fc_hidden_size = fc_hidden_size

		self.bert_wrapper = bert_wrapper
		self.bert_wrapper.bert_layer.trainable = bert_trainable
		self.batch_max_sentences = batch_max_sentences
		self.batch_max_windows = batch_max_windows

		self.init_model = init_model

		if optimizer == "adam":
			self.optimizer = keras.optimizers.Adam(lr=self.learning_rate)

		if loss == "categorical_crossentropy":
			self.loss = keras.losses.CategoricalCrossentropy(from_logits=False)

		self._build_model()  # builds self.model variable

	
	def _build_embedding_mask(self):
		embedding_mask_weights = np.ones((self.alphabet_size, self.num_of_classes))
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
		input_bert_ids = Input(shape=(self.batch_max_sentences, self.bert_wrapper.max_seq_len), name='bert_input_ids', dtype='int32')
		input_bert_seg = Input(shape=(self.batch_max_sentences, self.bert_wrapper.max_seq_len), name='bert_segment_ids', dtype='int32')
		input_token_ids = Input(shape=(self.batch_max_windows,), name='token_ids', dtype='int32')
		input_sent_ids = Input(shape=(self.batch_max_windows,), name='sent_ids', dtype='int32')
		input_mask = Input(shape=(self.batch_max_windows,), name='mask', dtype='float32')
		input_char_windows = Input(shape=(self.batch_max_windows, self.window_size), name='char_windows', dtype='int32')
		
		keras_internal_batch_size = K.shape(input_token_ids)[0]

		##########################################################################
		######################  Bert  ############################################
		input_bert_ids_reshaped = tf.reshape(input_bert_ids, shape=(-1, self.bert_wrapper.max_seq_len), name="reshape_input_bert_ids")
		input_bert_seg_reshaped = tf.reshape(input_bert_seg, shape=(-1, self.bert_wrapper.max_seq_len), name="reshape_input_bert_seg")
		# shape = (?batch_size x max_sent, max_seq_len)

		bert_output = self.bert_wrapper.bert_layer(input_bert_ids_reshaped, input_bert_seg_reshaped)
		# bert_output = (?batch_size x max_sent, bert_max_seq_len, bert_hidden_size)
		bert_output = tf.reshape(bert_output, shape=(-1, self.batch_max_sentences, self.bert_wrapper.max_seq_len, self.bert_wrapper.hidden_size), name="bert_output")
		# bert_output = (?batch_size, max_sent, bert_max_seq_len, bert_hidden_size)

		##########################################################################

		##########################################################################
		######################  CharCNN  #########################################
		embedding_mask_weights = self._build_embedding_mask()
		input_char_windows_reshaped = tf.reshape(input_char_windows, shape=(-1, self.window_size), name="reshape_input_char_windows")
		# shape = (?batch_size x max_windows, window_size)
		# char mask
		char_mask = Embedding(self.alphabet_size, self.num_of_classes, input_length=1, trainable=False, weights=[embedding_mask_weights], name="mask_embedding")(input_char_windows_reshaped[:, (self.window_size-1)//2])				
		char_mask = tf.reshape(char_mask,(-1, self.batch_max_windows, self.num_of_classes), name="reshape_char_mask")
		# Embedding layer
		x = Embedding(self.alphabet_size, self.embedding_size, input_length=self.window_size, trainable=True, name="sequence_embedding")(input_char_windows_reshaped)
		# x = (?batch_size, window_size, embedding_size)
		middle_char_embedding = x[:,(self.window_size-1)//2]

		# Convolution layers
		convolution_output = []
		for num_filters, filter_width in self.conv_layers:
			conv = Conv1D(filters=num_filters, kernel_size=filter_width, activation='tanh',
									name='Conv1D_{}_{}'.format(num_filters, filter_width))(x)
			# conv = (?batch_size, window_size-filter_size+1, num_filters)
			pool = GlobalMaxPooling1D(name='MaxPoolingOverTime_{}_{}'.format(num_filters, filter_width))(conv)
			# pool = (?batch_size, num_filters)
			convolution_output.append(pool)

		if convolution_output != []:
			x = Concatenate()(convolution_output)
			# x = (?batch_size, total_number_of_filters)
			x = Dropout(rate=self.cnn_dropout_rate)(x)

			# concatenate middle char
			x = Concatenate()([x, middle_char_embedding])

			self.total_number_of_filters = self.total_number_of_filters + self.embedding_size

		else:
			x = Flatten()(x)
			self.total_number_of_filters = self.window_size * self.embedding_size

		char_cnn_output = Dropout(rate=self.cnn_dropout_rate)(x)
		char_cnn_output = tf.reshape(char_cnn_output, shape=(-1, self.batch_max_windows, self.total_number_of_filters), name="char_cnn_output")
		# char_cnn_otput = (?batch_size, max_windows, total_filters)
		##########################################################################
		
		
		# get bert tokens coresponding to sent_ids and token_ids
		batch_indexes = tf.range(0, keras_internal_batch_size, name="range_batch_indexes")
		batch_indexes = tf.reshape(batch_indexes, (-1,1), name="reshape_batch_indexes")
		batch_indexes = tf.tile(batch_indexes, (1,self.batch_max_windows), name="tile_batch_indexes")
		indices = tf.stack([batch_indexes, input_sent_ids, input_token_ids], axis = 2)
		bert_tokens = tf.gather_nd(bert_output, indices, name="bert_tokens")
		# apply bert dropout here?
		# bert_tokens = (?batch_size, max_windows, bert_hidden_size)
		bert_cnn_concatenation = Concatenate()([bert_tokens, char_cnn_output])
		# bert_cnn_concatenation = char_cnn_output
		
		# hidden layer
		hidden = Dense(self.fc_hidden_size, activation='relu')(bert_cnn_concatenation)

		# Output layer
		predictions = Dense(self.num_of_classes, activation='softmax')(hidden)
		# mask predictions based on middle char 
		masked_predictions = keras.layers.multiply([predictions, char_mask])
		
		
		input_mask_reshaped = tf.reshape(input_mask, (-1, 1), name="reshape_input_mask")
		# mask prediction based on window mask
		# extended_mask = tf.reshape(input_mask, (-1, self.batch_max_windows, 1))
		# extended_mask = tf.tile(extended_mask, [1, 1, self.num_of_classes])	
		# masked_predictions = keras.layers.multiply([masked_predictions, extended_mask])
		
		flatten_masked_predictions = tf.reshape(masked_predictions, shape=(-1, self.num_of_classes), name="resh_flatmaskpred")
		# flatten_masked_predictions = masked_predictions
		# flatten_masked_prediction = (?batch_size x max_windows, num_of_classes)

		# Build and compile model
		model = Model(inputs=[input_bert_ids, input_bert_seg, input_token_ids, input_sent_ids, input_mask, input_char_windows], outputs=[flatten_masked_predictions, input_mask_reshaped])

		weights = np.ones(self.num_of_classes)
		model.compile(optimizer=self.optimizer, loss=[weighted_categorical_crossentropy(weights, self.num_of_classes).loss, None], metrics=[categorical_acc])

		if self.init_model != None:
			model.load_weights(self.init_model)
		else:
			self.bert_wrapper.load_weights()

		self.model = model
		# print("Bert+CharCNN model built: ")
		# self.model.summary()
		

	def train(self, train_dataset, train_batch_size, train_size, dev_dataset, dev_batch_size, dev_size, epochs, file_evalname, char_to_id_dict, model_filename):

		best_wa_dia = -1
		best_wa_all = -1
		best_ca_dia = -1
		best_ca_all = -1
		best_epoch = -1

		dev_steps = (dev_size // dev_batch_size) + 1
		
		if dev_batch_size == 1:
			dev_steps += 1

		for i in range(epochs):
			print("EPOCH ", (i+1))
			self.model.fit(train_dataset, steps_per_epoch=train_size//train_batch_size, epochs=1, verbose=1)
			wa_dia, wa_all, ca_dia, ca_all, _ = utils.evaluate_model(self.model, file_evalname, dev_dataset, dev_steps)
			if wa_dia > best_wa_dia:
				best_wa_dia = wa_dia
				best_wa_all = wa_all
				best_ca_dia = ca_dia
				best_ca_all = ca_all
				best_epoch = i+1
				self.model.save(model_filename+".h5")
			
			print("Best model: epoch =", best_epoch, "best word_accuracy_dia =", format(best_wa_dia, '.4f'), "best word_accuracy_all =", format(best_wa_all, '.4f'), 
							"best char_accuracy_dia =", format(best_ca_dia, '.4f'), "best char_accuracy_all =", format(best_ca_all, '.4f'))
			print("---------------")

def categorical_acc(y_true, y_pred):
	# TODO: change this to number of classes
	y_true = tf.reshape(y_true, shape=(-1, 5), name="reshape_acc")
	return keras.metrics.categorical_accuracy(y_true, y_pred)


class weighted_categorical_crossentropy(object):
	"""
	A weighted version of keras.objectives.categorical_crossentropy

	Variables:
		weights: numpy array of shape (C,) where C is the number of classes

	Usage:
		loss = weighted_categorical_crossentropy(weights).loss
		model.compile(loss=loss,optimizer='adam')
	"""

	def __init__(self,weights,num_of_classes):
		self.weights = K.variable(weights)
		self.num_of_classes = num_of_classes

        
	def loss(self, y_true, y_pred):
		y_true = tf.reshape(y_true, shape=(-1, self.num_of_classes), name="reshape_loss")
		
		# scale preds so that the class probas of each sample sum to 1
		y_pred = y_pred / K.sum(y_pred, axis=-1, keepdims=True)
		
		# clip
		y_pred = K.clip(y_pred, K.epsilon(), 1)
			
		# calc
		loss = y_true*K.log(y_pred)*self.weights
		loss =-K.sum(loss,-1)
		return loss

