#pylint: disable=import-error
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Concatenate, Convolution1D, GlobalMaxPooling1D, Embedding, Dropout, Flatten
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.callbacks import TensorBoard
import sys
import numpy as np
from tensorflow.keras import backend as K
import rb.processings.diacritics.utils as utils


class CharCNN(object):
	"""
	Class to implement simple Character Level Convolutional Neural Network
	The model is used to classify diacritics
    """


	def __init__(self, input_size, alphabet_size, embedding_size, conv_layers, num_of_classes, fc_hidden_size,
					cnn_dropout_rate, learning_rate, optimizer='adam', loss='categorical_crossentropy'):
		"""
        Initialization for the Character Level CNN model.
        Args:
            input_size (int): Size of input features
            alphabet_size (int): Size of alphabets to create embeddings for
            embedding_size (int): Size of embeddings
            conv_layers (list[list[int]]): List of Convolution layers for model
            num_of_classes (int): Number of classes in data
			fc_hidden_size (int) : Size of hidden layer (between conv features and prediction)
            dropout_rate (float): Dropout Rate
            optimizer (str): Training optimizer
            loss (str): Loss function
        """
		self.input_size = input_size
		self.alphabet_size = alphabet_size
		self.embedding_size = embedding_size
		self.conv_layers = conv_layers
		self.num_of_classes = num_of_classes
		self.fc_hidden_size = fc_hidden_size
		self.cnn_dropout_rate = cnn_dropout_rate
		self.learning_rate = learning_rate

		if optimizer == "adam":
			self.optimizer = keras.optimizers.Adam(lr=self.learning_rate)

		if loss == "categorical_crossentropy":
			self.loss = keras.losses.CategoricalCrossentropy(from_logits=False)

		self._build_model()  # builds self.model variable

	
	def _build_model(self):
		"""
        Build and compile the Character Level CNN model
        Returns: None
        """

		embedding_mask_weights = np.zeros((self.alphabet_size, self.num_of_classes))
		# a -> a, ă, â
		embedding_mask_weights[2] = [1,1,1,0,0]
		# s -> s, ș
		embedding_mask_weights[10] = [1,0,0,1,0]
		# t -> t, ț
		embedding_mask_weights[13] = [1,0,0,0,1]
		# i -> i, î
		embedding_mask_weights[16] = [1,0,1,0,0]


		# Input layer
		inputs = Input(shape=(self.input_size,), name='input_layer', dtype='int32')
		# mask
		mask = Embedding(self.alphabet_size, 5, input_length=1, trainable=False, weights=[embedding_mask_weights], name="mask_embedding")(inputs[:,(self.input_size-1)//2])
						
		# Embedding layer
		x = Embedding(self.alphabet_size, self.embedding_size, input_length=self.input_size, trainable=True, name="sequence_embedding")(inputs)
		# x = (?batch_size, window_size, embedding_size)
		middle_char_embedding = x[:,(self.input_size-1)//2]


		# Convolution layers
		convolution_output = []
		for num_filters, filter_width in self.conv_layers:
			conv = Conv1D(filters=num_filters, kernel_size=filter_width, activation='tanh',
									name='Conv1D_{}_{}'.format(num_filters, filter_width))(x)
			# conv = (?batch_size, window_size-filter_size+1, num_filters)
			pool = GlobalMaxPooling1D(name='MaxPoolingOverTime_{}_{}'.format(num_filters, filter_width))(conv)
			# conv = (?batch_size, num_filters)
			convolution_output.append(pool)


		if convolution_output != []:
			x = Concatenate()(convolution_output)
			# x = (?batch_size, total_number_of_filters)
			x = Dropout(rate=self.cnn_dropout_rate)(x)

			# concatenate middle char
			x = Concatenate()([x, middle_char_embedding])

		else:
			x = Flatten()(x)

		hidden_layer = Dense(self.fc_hidden_size, activation='relu')(x)


		# Output layer
		predictions = Dense(self.num_of_classes, activation='softmax')(hidden_layer)
		# mask predictions based on middle char
		masked_predictions = keras.layers.multiply([predictions, mask])
		# masked_predictions = predictions

		# Build and compile model
		model = Model(inputs=inputs, outputs=masked_predictions)

		# weights = np.ones(self.num_of_classes)
		# model.compile(optimizer=self.optimizer, loss=weighted_categorical_crossentropy(weights).loss, metrics=[tf.keras.metrics.categorical_accuracy])
		model.compile(optimizer=self.optimizer, loss=self.loss, metrics=[tf.keras.metrics.categorical_accuracy])
		self.model = model
		print("CharCNN model built: ")
		self.model.summary()
		# sys.exit()


	def train(self, train_dataset, train_batch_size, train_size, dev_dataset, dev_batch_size, dev_size, epochs, file_evalname, char_to_id_dict, model_filename):

		best_wa_dia = -1
		best_wa_all = -1
		best_ca_dia = -1
		best_ca_all = -1
		best_epoch = -1

		for i in range(epochs):
			print("EPOCH ", (i+1))
			self.model.fit(train_dataset, steps_per_epoch=train_size//train_batch_size, epochs=1, verbose=1)
			# self.model.evaluate(dev_dataset, steps=dev_size//dev_batch_size, verbose=1)
			# print("---------------")
			# wa_dia, wa_all, ca_dia, ca_all, _ = utils.evaluate_model_on_file(self.model, file_evalname, char_to_id_dict, self.input_size)
			wa_dia, wa_all, ca_dia, ca_all, _ = utils.evaluate_model(self.model, file_evalname, dev_dataset, (dev_size//dev_batch_size)+1)
			if wa_dia > best_wa_dia:
				best_wa_dia = wa_dia
				best_wa_all = wa_all
				best_ca_dia = ca_dia
				best_ca_all = ca_all
				best_epoch = i+1
				self.model.save(model_filename+'.h5')
				
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
