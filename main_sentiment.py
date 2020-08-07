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
import sys
from rb.core.lang import Lang
from rb.processings.sentiment.BertRegression import BertRegression
import rb.processings.sentiment.utils_new as utils
from rb.processings.encoders.bert import BertWrapper

import numpy as np
from tensorflow.keras.models import load_model

from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)


FLAGS = absl.flags.FLAGS
# model
absl.flags.DEFINE_integer('bert_max_seq_len', 128, 'Maximum sequence length')
absl.flags.DEFINE_string('bert_model_type', "base", "BERT model type: small, base, large or multi_cased_base")
absl.flags.DEFINE_string('bert_pooling_type', "cls", "BERT pooling type: cls, pool or cnn")
absl.flags.DEFINE_boolean('bert_trainable', True, 'Whether BERT part of model is trainable or not')
absl.flags.DEFINE_list('hidden_size', "256", "List of sizes of fc hidden layer or cnn width (e.g. 256, 128, 32)")
absl.flags.DEFINE_float('dropout_rate', 0.1, 'Dropout rate')
# training details
absl.flags.DEFINE_string('optimizer', 'adam', 'Optimizer used for training')
absl.flags.DEFINE_string('loss', 'mse', 'Loss used for training')
absl.flags.DEFINE_float('learning_rate', 1e-5, 'Learning Rate used for optimization')
absl.flags.DEFINE_integer('batch_size', 64, 'Batch size to use during training')
absl.flags.DEFINE_integer('epochs', 20, 'Number of epochs to train')
# saving or restoring
absl.flags.DEFINE_string('restore_model', None, 'Initial model to start training/evaluating from')
absl.flags.DEFINE_string('experiment_name', "0", 'Name of current experiment. Used for saving model')
absl.flags.DEFINE_string('models_path',"rb/processings/sentiment/models/", "Models path for saving and restoring")
# misc
absl.flags.DEFINE_string('run_type',"train", "Whether to train model or evaluate model")
absl.flags.DEFINE_string('dataset_folder_path', 'rb/processings/sentiment/corpus/reallife/', 'Path to dataset folder')
# absl.flags.DEFINE_string('dataset_folder_path', 'rb/processings/sentiment/corpus/original/', 'Path to dataset folder')

def getDataset(split, batch_size, bert_wrapper):
	data = utils.readJson(FLAGS.dataset_folder_path+"{0}.json".format(split))
	features = utils.getFeatures(data)
	features, labels, weights_dict = utils.processFeatures(features, bert_wrapper)
	print(split, len(features[0]), len(labels), weights_dict)

	# go to tf.Data format
	t1 = []
	t2 = []
	for i in range(len(features[0])):
		t1.append(features[0][i])
		t2.append(features[1][i])
	dataset = tf.data.Dataset.from_tensor_slices(((t1, t2), labels))
	if split == "train":
		dataset = dataset.shuffle(700000, reshuffle_each_iteration=True)
	dataset = dataset.batch(FLAGS.batch_size)
	return dataset

def main(argv):

	del argv
	# utils.build_reallife_corpus(FLAGS.dataset_folder_path)
	# sys.exit()

	bert_wrapper = BertWrapper(Lang.RO, max_seq_len=FLAGS.bert_max_seq_len, model_name=FLAGS.bert_model_type)
	model = BertRegression(bert_wrapper=bert_wrapper, bert_trainable=FLAGS.bert_trainable, bert_pooling_type=FLAGS.bert_pooling_type,
					learning_rate=FLAGS.learning_rate, hidden_size=FLAGS.hidden_size, restore_model=FLAGS.restore_model,
					optimizer=FLAGS.optimizer, loss=FLAGS.loss, dropout_rate=FLAGS.dropout_rate, models_path=FLAGS.models_path)

	if FLAGS.run_type == "train":
		train_dataset = getDataset("train", FLAGS.batch_size, bert_wrapper)
		dev_dataset = getDataset("dev", FLAGS.batch_size, bert_wrapper)
		model.train(train_dataset, dev_dataset, FLAGS.epochs, FLAGS.experiment_name)

	if FLAGS.run_type == "eval":
		dev_dataset = getDataset("dev", FLAGS.batch_size, bert_wrapper)
		test_dataset = getDataset("test", FLAGS.batch_size, bert_wrapper)
		dev_loss = model.eval(dev_dataset)
		test_loss = model.eval(test_dataset)
		print("dev =", format(dev_loss, '.4f'), "test =", format(test_loss, '.4f'))

if __name__ == "__main__":
	absl.app.run(main)