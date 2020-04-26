import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    except RuntimeError as e:
        print(e)
import pickle
import absl
import rb.processings.diacritics.utils as utils
from rb.processings.diacritics.CharCNN import CharCNN
from rb.processings.diacritics.BertCNN import BertCNN
import sys
# import bert
# from bert.tokenization.bert_tokenization import FullTokenizer
from rb.core.lang import Lang
from rb.processings.encoders.bert import BertWrapper
from tensorflow.keras.models import load_model
from rb.processings.diacritics.BertCNN import weighted_categorical_crossentropy, categorical_acc
import numpy as np




FLAGS = absl.flags.FLAGS
absl.flags.DEFINE_string('dataset_folder_path', 'rb/processings/diacritics/dataset/split/', 'Path to dataset folder')
absl.flags.DEFINE_integer('window_size', 11, "Character total window size (left + center + right)")
absl.flags.DEFINE_integer('train_batch_size', 16, "Batch size to be used for training")
absl.flags.DEFINE_integer('dev_batch_size', 64, "Batch size to be used for evaluation")
absl.flags.DEFINE_integer('char_embedding_size', 50, "Dimension of character embedding")
absl.flags.DEFINE_integer("cnn_filter_size", 50, "Size of cnn filters (it applies the same size to all filters)")
absl.flags.DEFINE_integer("fc_hidden_size", 128, "Size of fc hidden layer (between features and predictions)")
absl.flags.DEFINE_integer('epochs', 20, "Number of epochs to train")
absl.flags.DEFINE_float('learning_rate', 1e-3, "Learning rate")
absl.flags.DEFINE_string('optimizer', 'adam', 'Optimizer')
absl.flags.DEFINE_float('dropout_rate', 0.1, "Dropout rate: fraction of units to drop during training")
absl.flags.DEFINE_string('model_type', 'BertCNN', "Type of model: CharCNN or BertCNN")
absl.flags.DEFINE_string('model_filename', 'modelX', 'Name of model to save')

absl.flags.DEFINE_string('bert_model_type', "small", "BERT model type: small, base, large or multi_cased_base")
absl.flags.DEFINE_boolean('bert_trainable', False, 'Whether to BERT is trainable or not')
absl.flags.DEFINE_string('init_model', None, 'Initial model to start training from')
absl.flags.DEFINE_integer('bert_max_seq_len', 128, "Maximum sequence length for BERT models")
absl.flags.DEFINE_integer('batch_max_sentences', 10, "Maximum sentences per batch")
absl.flags.DEFINE_integer('batch_max_windows', 280, "Maximum windows per batch")
absl.flags.DEFINE_integer('no_classes', 5, "Number of classes for clasification")

absl.flags.DEFINE_integer('dev_version', 1, "0 for 2-3-4-5, 1 for 2-3-4-5-11, 2 for simple FC")

def main(argv):

	del argv

	# load dict
	char_dict = pickle.load(open(FLAGS.dataset_folder_path+"char_dict", "rb"))
	
	# build_cnn_filters
	conv_layers = []
	if FLAGS.dev_version == 0:
		filters = [2,3,4,5]
	elif FLAGS.dev_version == 1:
		filters = [2,3,4,5,11]

	elif FLAGS.dev_version == 2:
		filters = []

	for i in filters:
		conv_layers.append([FLAGS.cnn_filter_size, i])
	
	if FLAGS.model_type == "CharCNN":

		train_dataset = tf.data.Dataset.from_generator(lambda : utils.generator_cnn_features(FLAGS.dataset_folder_path+"train.txt", char_dict, FLAGS.window_size),
						output_types = (tf.int32, tf.float32), output_shapes=([FLAGS.window_size], [FLAGS.no_classes]))
		train_dataset = train_dataset.shuffle(int(1e5), reshuffle_each_iteration=True).batch(FLAGS.train_batch_size).repeat(-1)
		# train number of features 61640402 | train number of sentences = 2126626	
		train_size = 61640402
		
		dev_dataset = tf.data.Dataset.from_generator(lambda : utils.generator_cnn_features(FLAGS.dataset_folder_path+"dev.txt", char_dict, FLAGS.window_size),
						output_types = (tf.int32, tf.float32), output_shapes=([FLAGS.window_size], [FLAGS.no_classes]))
		dev_dataset = dev_dataset.batch(FLAGS.dev_batch_size)
		# dev number of features 6838456 | dev number of sentences = 236288	
		dev_size = 6838456
		
		# test number of features 3613915 | dev number of sentences = N/A
		# test_size = 3613915

		model = CharCNN(input_size=FLAGS.window_size, alphabet_size=len(char_dict), conv_layers = conv_layers, fc_hidden_size = FLAGS.fc_hidden_size,
							embedding_size=FLAGS.char_embedding_size, num_of_classes=FLAGS.no_classes, cnn_dropout_rate=FLAGS.dropout_rate, learning_rate=FLAGS.learning_rate)

		model.train(train_dataset, FLAGS.train_batch_size, train_size//1, dev_dataset, FLAGS.dev_batch_size, dev_size//1, 
							FLAGS.epochs, "rb/processings/diacritics/dataset/split/dev.txt", char_dict, FLAGS.model_filename)
		

	elif FLAGS.model_type == "BertCNN":
		
		bert_wrapper = BertWrapper(Lang.RO, max_seq_len=FLAGS.bert_max_seq_len, model_name=FLAGS.bert_model_type)
		train_dataset = tf.data.Dataset.from_generator(lambda : utils.generator_bert_cnn_features(FLAGS.dataset_folder_path+"train.txt", char_dict, FLAGS.window_size, bert_wrapper, FLAGS.batch_max_sentences, FLAGS.batch_max_windows),
						output_types=({'bert_input_ids': tf.int32, 'bert_segment_ids': tf.int32, 'token_ids': tf.int32, 'sent_ids': tf.int32,
										'mask': tf.float32, 'char_windows': tf.int32}, tf.float32),
						output_shapes=({'bert_input_ids':[FLAGS.batch_max_sentences, FLAGS.bert_max_seq_len], 'bert_segment_ids':[FLAGS.batch_max_sentences, FLAGS.bert_max_seq_len], 'token_ids':[FLAGS.batch_max_windows],
										'sent_ids': [FLAGS.batch_max_windows], 'mask': [FLAGS.batch_max_windows], 'char_windows': [FLAGS.batch_max_windows, FLAGS.window_size]}, [FLAGS.batch_max_windows, 5]))
		train_dataset = train_dataset.shuffle(int(1e3), reshuffle_each_iteration=True).batch(FLAGS.train_batch_size).repeat(-1)
		# max_sent = 10, max_windows = 280
		train_size = 243954
		
		dev_dataset = tf.data.Dataset.from_generator(lambda : utils.generator_bert_cnn_features(FLAGS.dataset_folder_path+"dev.txt", char_dict, FLAGS.window_size, bert_wrapper, FLAGS.batch_max_sentences, FLAGS.batch_max_windows),
						output_types=({'bert_input_ids': tf.int32, 'bert_segment_ids': tf.int32, 'token_ids': tf.int32, 'sent_ids': tf.int32,
										'mask': tf.float32, 'char_windows': tf.int32}, tf.float32),
						output_shapes=({'bert_input_ids':[FLAGS.batch_max_sentences, FLAGS.bert_max_seq_len], 'bert_segment_ids':[FLAGS.batch_max_sentences, FLAGS.bert_max_seq_len], 'token_ids':[FLAGS.batch_max_windows],
										'sent_ids': [FLAGS.batch_max_windows], 'mask': [FLAGS.batch_max_windows], 'char_windows': [FLAGS.batch_max_windows, FLAGS.window_size]}, [FLAGS.batch_max_windows, FLAGS.no_classes]))
		dev_dataset = dev_dataset.batch(FLAGS.dev_batch_size)
		# max_sent = 10, max_windows = 280
		dev_size = 27100

		model = BertCNN(window_size=FLAGS.window_size, alphabet_size=len(char_dict), conv_layers = conv_layers, fc_hidden_size = FLAGS.fc_hidden_size,
					embedding_size=FLAGS.char_embedding_size, num_of_classes=FLAGS.no_classes, batch_max_sentences=FLAGS.batch_max_sentences, batch_max_windows=FLAGS.batch_max_windows,
					bert_wrapper=bert_wrapper, bert_trainable=FLAGS.bert_trainable, cnn_dropout_rate=FLAGS.dropout_rate, learning_rate=FLAGS.learning_rate, init_model=FLAGS.init_model)

		# TODO: make this automatic
		model_save_path = "rb/processings/diacritics/models/bert_models/" + FLAGS.model_filename

		model.train(train_dataset, FLAGS.train_batch_size, train_size//1, dev_dataset, FLAGS.dev_batch_size, dev_size//1, 
							FLAGS.epochs, "rb/processings/diacritics/dataset/split/dev.txt", char_dict, model_save_path)

if __name__ == "__main__":
	absl.app.run(main)