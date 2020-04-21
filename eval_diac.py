#pylint: disable=import-error
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
from rb.core.lang import Lang
from rb.processings.encoders.bert import BertWrapper
import os
from tensorflow.keras.models import load_model


FLAGS = absl.flags.FLAGS
absl.flags.DEFINE_string('folder_path', 'rb/processings/diacritics', 'Path to base folder where we have the model and the dataset')
absl.flags.DEFINE_string('model_name', 'model.h5', 'Model name')
absl.flags.DEFINE_string('model_type', 'CharCNN', "Type of model: CharCNN or BertCNN")
absl.flags.DEFINE_integer('batch_size', 1024, "Batch size to be used for evaluation")

# absl.flags.DEFINE_string('dataset_folder_path', 'rb/processings/diacritics/dataset/split/', 'Path to bert dataset folder')
# absl.flags.DEFINE_integer('window_size', 11, "Character total window size (left + center + right)")
# absl.flags.DEFINE_integer('train_batch_size', 1024, "Batch size to be used for training")
# absl.flags.DEFINE_integer('dev_batch_size', 512, "Batch size to be used for evaluation")
# absl.flags.DEFINE_integer('char_embedding_size', 100, "Dimension of character embedding")
# absl.flags.DEFINE_integer("cnn_filter_size", 10, "Size of cnn filters (it applies the same size to all filters)")
# absl.flags.DEFINE_integer('epochs', 25, "Number of epochs to train")
# absl.flags.DEFINE_float('learning_rate', 1e-4, "Learning rate")
# absl.flags.DEFINE_string('optimizer', 'adam', 'Optimizer')
# absl.flags.DEFINE_float('dropout_rate', 0.1, "Dropout rate: fraction of units to drop during training")
# absl.flags.DEFINE_string('model_type', 'CharCNN', "Type of model: CharCNN or BertCNN")

# absl.flags.DEFINE_string('bert_model_dir', "models/bert_models/ro0/", "Path to folder where BERT model is located")
# absl.flags.DEFINE_boolean('bert_trainable', False, 'Whether to BERT is trainable or not')
# absl.flags.DEFINE_integer('bert_max_seq_len', 128, "Maximum sequence length for BERT models")
# absl.flags.DEFINE_integer('batch_max_sentences', 10, "Maximum sentences per batch")
# absl.flags.DEFINE_integer('batch_max_windows', 280, "Maximum windows per batch")
# absl.flags.DEFINE_integer('no_classes', 5, "Number of classes for clasification")


def main(argv):

	del argv
	
	dataset_path = os.path.join(FLAGS.folder_path, "dataset/split/")
	dataset_dict = os.path.join(dataset_path, "char_dict")
	char_dict = pickle.load(open(dataset_dict, "rb"))

	dev_path = os.path.join(dataset_path, "dev.txt")
	test_path = os.path.join(dataset_path, "test.txt")


	if FLAGS.model_type == "CharCNN":
		models_path = os.path.join(FLAGS.folder_path, "models/char_cnn_models/")
		# dev and test generator
		dev_dataset = tf.data.Dataset.from_generator(lambda : utils.generator_cnn_features(dev_path, char_dict, 11),
							output_types = (tf.int32, tf.float32), output_shapes=([11], [5]))
		dev_dataset = dev_dataset.batch(FLAGS.batch_size, drop_remainder=False)
		dev_size = 6838456

		test_dataset = tf.data.Dataset.from_generator(lambda : utils.generator_cnn_features(test_path, char_dict, 11),
							output_types = (tf.int32, tf.float32), output_shapes=([11], [5]))
		test_dataset = test_dataset.batch(FLAGS.batch_size, drop_remainder=False)
		test_size = 3613915


	elif FLAGS.model_type == "BertCNN":
		# dev and test generator
		models_path = os.path.join(FLAGS.folder_path, "models/bert_models/")
	
	model_path = os.path.join(models_path, FLAGS.model_name)


	model = load_model(model_path)
	# evaluate on dev
	utils.evaluate_model(model, dev_path, dev_dataset, (dev_size//FLAGS.batch_size)+1, write_to_file=True, outfile_name=model_path.split(".")[0]+"_dev_out.txt")

	# evaluate on test
	utils.evaluate_model(model, test_path, test_dataset, (test_size//FLAGS.batch_size)+1, write_to_file=True, outfile_name=model_path.split(".")[0]+"_test_out.txt")

if __name__ == "__main__":
	absl.app.run(main)