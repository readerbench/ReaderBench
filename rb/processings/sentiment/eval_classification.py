import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    except RuntimeError as e:
        print(e)
import tensorflow.keras as keras
import numpy as np
import BertModel
import utils
import sys
import absl
import bert
import functools
import os


FLAGS = absl.flags.FLAGS

absl.flags.DEFINE_integer('max_seq_len', 128, 'Maximum sequence length')
absl.flags.DEFINE_string('model_folder_path', '../Models/raw/multi_cased_base/', 'Path to bert model folder')
absl.flags.DEFINE_string('model_weights_path', 'models/finetune_sentiment_emag/multi_cased_base/2/model.h5', "Path to bert finetuned weights")
absl.flags.DEFINE_integer('dev_batch_size', 32, 'Batch size to use during evaluation on dev')
absl.flags.DEFINE_integer('test_batch_size', 32, 'Batch size to use during evaluation on test')
absl.flags.DEFINE_integer('num_classes', 4, "Number of classes for clasification task")
absl.flags.DEFINE_bool("use_tpu", False, "Use TPU or not")
absl.flags.DEFINE_string("tpu_name", None, "Name of TPU instance")


def create_model():
    # define input
    input_ids = tf.keras.layers.Input(shape=(FLAGS.max_seq_len), dtype=tf.int32, name="input_ids")
    # no need for attention mask; it is computed in automatically in bert model
    # attention_mask = tf.keras.layers.Input(shape=(FLAGS.max_seq_len), dtype=tf.int32, name="input_ids")
    segment_ids = tf.keras.layers.Input(shape=(FLAGS.max_seq_len), dtype=tf.int32, name="segment_ids")

    do_lower_case = True
    if "multi_cased_base" in FLAGS.model_folder_path:
        do_lower_case = False

    # define model
    bert_model = BertModel.BertModel(FLAGS.model_folder_path, FLAGS.max_seq_len, do_lower_case)
    bert_output = bert_model.bert_layer([input_ids, segment_ids])
    cls_output = keras.layers.Lambda(lambda seq: seq[:, 0, :])(bert_output)
    fc1 = keras.layers.Dense(units=100, activation="relu")(cls_output)
    prediction = keras.layers.Dense(units=FLAGS.num_classes, activation="softmax")(fc1)

    # build model
    model = keras.Model(inputs=[input_ids, segment_ids], outputs=prediction)
    model.build(input_shape=[(None, FLAGS.max_seq_len), (None, FLAGS.max_seq_len)])
    # load pretrained
    model.load_weights(FLAGS.model_weights_path)
    
    model.compile(loss = keras.losses.CategoricalCrossentropy(from_logits=False), metrics = [tf.keras.metrics.categorical_accuracy])
    model.summary()

    print("Do lower case =", do_lower_case)


    return model, bert_model


def create_tfdataset(features, labels, split, batch_size):

    aux1 = []
    aux2 = []
    for i in range(len(features[0])):
        aux1.append(features[0][i])
        aux2.append(features[1][i])

    dataset = tf.data.Dataset.from_tensor_slices(((aux1, aux2), labels))
    if split == "train":
        dataset = dataset.shuffle(22000)
        dataset = dataset.repeat(-1).batch(batch_size)

    else:
        dataset = dataset.batch(batch_size)

    return dataset

def main(argv):
    del argv
    
    # create model
    if FLAGS.use_tpu == True:
        tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(FLAGS.tpu_name, zone=None, project=None)
        tf.config.experimental_connect_to_cluster(tpu_cluster_resolver)
        tf.tpu.experimental.initialize_tpu_system(tpu_cluster_resolver)
        strategy = tf.distribute.experimental.TPUStrategy(tpu_cluster_resolver)
        with strategy.scope():
            model, bert = create_model()
    else:
            model, bert = create_model()
    
    # load data
    dev_data = utils.readJson("../Dataset/Reviews/4Classes/dev.json")
    test_data = utils.readJson("../Dataset/Reviews/4Classes/test.json")

    dev_features = utils.getFeatures(dev_data)
    dev_features, dev_labels, _ = utils.processFeatures(dev_features, bert)
    print(len(dev_features[0]), len(dev_labels))

    test_features = utils.getFeatures(test_data)
    test_features, test_labels, _ = utils.processFeatures(test_features, bert)
    print(len(test_features[0]), len(test_labels))
    
    dev_dataset = create_tfdataset(dev_features, dev_labels, "dev", FLAGS.dev_batch_size)
    test_dataset = create_tfdataset(test_features, test_labels, "test", FLAGS.test_batch_size)

    fmetric_dev = utils.FScoreCallback(dev_dataset, len(dev_labels)//FLAGS.dev_batch_size, dev_labels)
    model.evaluate(dev_dataset, callbacks=[fmetric_dev])

    fmetric_test = utils.FScoreCallback(test_dataset, len(test_labels)//FLAGS.test_batch_size, test_labels)
    model.evaluate(test_dataset, callbacks=[fmetric_test])

if __name__ == "__main__":

    absl.flags.mark_flag_as_required('model_folder_path')
    absl.app.run(main)
