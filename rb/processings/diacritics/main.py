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
import utils
from CharCNN import CharCNN
import sys
import bert
from bert.tokenization import FullTokenizer



FLAGS = absl.flags.FLAGS
absl.flags.DEFINE_string('dataset_folder_path', 'dataset/split/', 'Path to bert dataset folder')
absl.flags.DEFINE_integer('window_size', 11, "Character total window size (left + center + right)")
absl.flags.DEFINE_integer('train_batch_size', 1, "Batch size to be used for training")
absl.flags.DEFINE_integer('dev_batch_size', 512, "Batch size to be used for evaluation")
absl.flags.DEFINE_integer('char_embedding_size', 100, "Dimension of character embedding")
absl.flags.DEFINE_integer('epochs', 10, "Number of epochs to train")
absl.flags.DEFINE_float('learning_rate', 1e-4, "Learning rate")
absl.flags.DEFINE_string('optimizer', 'adam', 'Optimizer')
absl.flags.DEFINE_float('dropout_rate', 0.2, "Dropout rate: fraction of units to drop during training")
absl.flags.DEFINE_string('model_type', 'CharCNN', "Type of model: CharCNN or BertCNN")
absl.flags.DEFINE_string('bert_model_dir', "models/bert_models/ro0/", "Path to folder where BERT model is located")


def main(argv):

	del argv

	# load dict
	char_dict = pickle.load(open(FLAGS.dataset_folder_path+"char_dict", "rb"))
	print(char_dict)
	
	if FLAGS.model_type == "CharCNN":

		train_dataset = tf.data.Dataset.from_generator(lambda : utils.generator_cnn_features(FLAGS.dataset_folder_path+"train.txt", char_dict, FLAGS.window_size),
						output_types = (tf.int32, tf.float32), output_shapes=([FLAGS.window_size], [5]))
		# train_dataset = train_dataset.shuffle(int(1e5), reshuffle_each_iteration=True).batch(FLAGS.train_batch_size).repeat(-1)
		# train_dataset = train_dataset.batch(FLAGS.train_batch_size)
		train_dataset = train_dataset.batch(FLAGS.train_batch_size)#.prefetch(tf.data.experimental.AUTOTUNE)

		# total number of features
		train_size = 61640402
		# total number of sentences
		# train_size = 2126626

		for index, _ in enumerate(train_dataset):
			if index % 1e6 == 0:
				print(index)
		
		sys.exit()
		
		dev_dataset = tf.data.Dataset.from_generator(lambda : utils.generator_cnn_features(FLAGS.dataset_folder_path+"dev.txt", char_dict, FLAGS.window_size),
						output_types = (tf.int32, tf.float32), output_shapes=([FLAGS.window_size], [5]))
		dev_dataset = dev_dataset.batch(FLAGS.dev_batch_size)
		# total number of features
		dev_size = 6838456
		# total number of sentences
		# dev_size = 236288

		# total number of features
		# test_size = 3613915

		model = CharCNN(input_size=FLAGS.window_size, alphabet_size=len(char_dict), conv_layers = [[20,1], [20,2], [20,3], [20,4], [20,5], [20,6], [20,7], [20,8], [20,9], [20,10]],
						embedding_size=FLAGS.char_embedding_size, num_of_classes=5,	dropout_rate=FLAGS.dropout_rate, learning_rate=FLAGS.learning_rate)

		model.train(train_dataset, FLAGS.train_batch_size, train_size//1, dev_dataset, FLAGS.dev_batch_size, dev_size//1, FLAGS.epochs, "dataset/split/dev.txt", char_dict)
		

	elif FLAGS.model_type == "BertCNN":
		
		# load bert_tokenizer
		bert_tokenizer = FullTokenizer(vocab_file=FLAGS.bert_model_dir+"vocab.vocab")
		print(bert_tokenizer)
		bert_tokenizer.basic_tokenizer._run_strip_accents = lambda x: x

		a = utils.generator_bert_cnn_features(FLAGS.dataset_folder_path+"train.txt", char_dict, FLAGS.window_size, bert_tokenizer)
		for x in a:
			print(x)
			# sys.exit()



if __name__ == "__main__":
	absl.app.run(main)