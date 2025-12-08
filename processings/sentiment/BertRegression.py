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
import os

class BertRegression(object):
	"""
	Class to implement simple Bert + Regression for sentiment analysis
    """
	def __init__(self, hidden_size, bert_trainable, bert_wrapper, bert_pooling_type,
				 dropout_rate, learning_rate, restore_model, optimizer, loss, models_path=None):
		"""
        Initialization for whole model
        Args:
			BERT side
			bert_wrapper (obj): Bert wrapper
			bert_trainable (bool): Whether to train BERT or not
			bert_pooling_type (str): Extraction mode from BERT sequence output. Either cls or pool 
			
			
			After BERT
			hidden_size ([int]): Sizes of hidden layer/cnn width after BERT
			dropout_rate (float): dropout between FC layers
			

			Learning
			optimizer (str): Training optimizer
            loss (str): Loss function
			learning_rate (float): Learning rate to use for training

			models_path (string): Path to models for saving and restoring
			restore_model (string): Name of model to restore for training/evaluating

        """
		
		self.learning_rate = learning_rate
		# process list
		self.hidden_size = list(map(lambda x: int(x), hidden_size))
		self.dropout_rate = dropout_rate

		self.bert_wrapper = bert_wrapper
		self.bert_wrapper.bert_layer.trainable = bert_trainable
		self.bert_pooling_type = bert_pooling_type

		self.models_path = models_path
		self.restore_model = restore_model

		if optimizer == "adam":
			self.optimizer = keras.optimizers.Adam(lr=self.learning_rate)
		elif optimizer == "sgd":
			self.optimizer = keras.optimizers.SGD(lr=self.learning_rate)

		if loss == "mse":
			self.loss = keras.losses.MeanSquaredError()

		self._build_model()  # builds self.model variable


	def _build_model(self):
		"""
        Build and compile the Bert + Character Level CNN model
        Returns: None
        """

		###########################  Bert  ####################################
		inputs, bert_output = self.bert_wrapper.create_inputs_and_model()
		convolution_output = []
		if self.bert_pooling_type == "cnn":
			num_filters = 256//len(self.hidden_size)
			for filter_width in self.hidden_size:
				conv = Conv1D(filters=num_filters, kernel_size=filter_width, activation='tanh',
								name='Conv1D_{}_{}'.format(num_filters, filter_width))(bert_output)
				pool = GlobalMaxPooling1D(name='MaxPoolingOverTime_{}_{}'.format(num_filters, filter_width))(conv)
				convolution_output.append(pool)
			bert_output = Concatenate()(convolution_output)
		else:
			bert_output = self.bert_wrapper.get_output(bert_output, self.bert_pooling_type)

		bert_output = Dropout(rate=self.dropout_rate)(bert_output)


		#####################  FC layers  #################################
		hidden = bert_output
		if self.bert_pooling_type == "cnn":
			hidden = Dense(128, activation='relu')(hidden)
		else:
			for layer_size in self.hidden_size:
				hidden = Dense(layer_size, activation='relu')(hidden)


		######################  Final prediction  ############################
		prediction = Dense(1, activation=None)(hidden)
		
		model = Model(inputs=inputs, outputs=[prediction])
		model.compile(optimizer=self.optimizer, loss=self.loss)
		
		if self.restore_model != None:
			model_restore_path = os.path.join(self.models_path, self.restore_model)
			loaded_model = tf.keras.models.load_model(model_restore_path, compile=False)
			weights = [layer.get_weights() for layer in loaded_model.layers]
			for layer, weight in zip(model.layers, weights):
				layer.set_weights(weight)

		else:
			self.bert_wrapper.load_weights()

		self.model = model
		# print("BERT model built: ")
		# self.model.summary()

	def train(self, train_dataset, dev_dataset, epochs, model_name):

		model_save_path = os.path.join(self.models_path, model_name)
		best_metric = 99999
		best_epoch = 0
		for i in range(epochs):
			print("EPOCH ", (i+1))
			self.model.fit(train_dataset, epochs=1, verbose=1)
			dev_loss = self.model.evaluate(dev_dataset)
			if dev_loss < best_metric:
				best_metric = dev_loss
				best_epoch = i+1
				self.model.save(model_save_path)
			print("Best model: epoch =", best_epoch, "best loss = ", format(best_metric, '.4f'))
			print("---------------")


	def eval(self, eval_dataset):
		eval_loss = self.model.evaluate(eval_dataset)
		return eval_loss