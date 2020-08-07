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
# TODO: change default value to None
absl.flags.DEFINE_string('model_folder_path', '../Models/raw_old/full/clean/trained_128_512/ro0/', 'Path to bert model folder')
absl.flags.DEFINE_float('learning_rate', 1e-5, 'Learning Rate used for optimization')
absl.flags.DEFINE_integer('batch_size', 32, 'Batch size to use during training')
absl.flags.DEFINE_integer('epochs', 1, 'Number of epochs to train')
absl.flags.DEFINE_float('dropout_rate', 0.5, 'Dropout rate')
absl.flags.DEFINE_integer('num_classes', 4, "Number of classes for clasification task")
absl.flags.DEFINE_integer('experiment_index', 1, 'Index of current experiment. Will be appended to weights file')
absl.flags.DEFINE_string('save_folder_path',".", "Save folder prefix")
absl.flags.DEFINE_bool("use_tpu", False, "Use TPU or not")
absl.flags.DEFINE_string("tpu_name", None, "Name of TPU instance")

def gen_data(features, labels):
    for index in range(len(features[0])):
        yield ({'input_ids': features[0][index], 'segment_ids': features[1][index]}, labels[index])


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
    cls_drop = keras.layers.Dropout(FLAGS.dropout_rate)(cls_output)
    fc1 = keras.layers.Dense(units=100, activation="relu")(cls_drop)
    prediction = keras.layers.Dense(units=FLAGS.num_classes, activation="softmax")(fc1)

    # build model
    model = keras.Model(inputs=[input_ids, segment_ids], outputs=prediction)
    model.build(input_shape=[(None, FLAGS.max_seq_len), (None, FLAGS.max_seq_len)])
    # load pretrained
    bert.load_bert_weights(bert_model.bert_layer, FLAGS.model_folder_path+"bert_model.ckpt")

    model.compile(optimizer=keras.optimizers.Adam(lr=FLAGS.learning_rate), loss = 'categorical_crossentropy', metrics = [tf.keras.metrics.categorical_accuracy])
    model.summary()

    print("Do lower case =", do_lower_case)

    return model, bert_model


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
    train_data = utils.readJson("../Dataset/Reviews/4Classes/train.json")
    dev_data = utils.readJson("../Dataset/Reviews/4Classes/dev.json")
    # test_data = utils.readJson("../Dataset/Reviews/4Classes/test.json")

    train_features = utils.getFeatures(train_data, sample_majority=True, sample_count=20000, seed=12345)
    train_features, train_labels, train_weights_dict = utils.processFeatures(train_features, bert)
    print(len(train_features[0]), len(train_labels), train_weights_dict)

    dev_features = utils.getFeatures(dev_data)
    dev_features, dev_labels, _ = utils.processFeatures(dev_features, bert)
    print(len(dev_features[0]), len(dev_labels))

    # test_features = utils.getFeatures(test_data)
    # test_features, test_labels, _ = utils.processFeatures(test_features, bert)
    # print(len(test_features[0]), len(test_labels))
    
    t1 = []
    t2 = []
    for i in range(len(train_features[0])):
        t1.append(train_features[0][i])
        t2.append(train_features[1][i])
    train_dataset = tf.data.Dataset.from_tensor_slices(((t1, t2), train_labels))
    train_dataset = train_dataset.shuffle(70000)
    train_dataset = train_dataset.repeat(-1).batch(FLAGS.batch_size)

    d1 = []
    d2 = []
    for i in range(len(dev_features[0])):
        d1.append(dev_features[0][i])
        d2.append(dev_features[1][i])
    dev_dataset = tf.data.Dataset.from_tensor_slices(((d1, d2), dev_labels))
    dev_dataset = dev_dataset.batch(32)

    # dataset for metric :(
    # dev_dataset_metric = tf.data.Dataset.from_generator(functools.partial(gen_data, dev_features, dev_labels), ({'input_ids': tf.int32, 'segment_ids': tf.int32},  tf.int32),
    #         ({'input_ids': tf.TensorShape([FLAGS.max_seq_len]), 'segment_ids': tf.TensorShape([FLAGS.max_seq_len])}, tf.TensorShape([None])))
    # dev_dataset_metric = dev_dataset_metric.batch(FLAGS.batch_size)
    fmetric = utils.FScoreCallback(dev_dataset, len(dev_labels)//32, dev_labels)


    folder_name = FLAGS.model_folder_path.split("/")[-2]+"_"+str(FLAGS.experiment_index)
    os.makedirs(FLAGS.save_folder_path+"/{0}/".format(folder_name))

    results = []
    for i in range(FLAGS.epochs):
        print("EPOCH ", i+1)
        _= model.fit(train_dataset, steps_per_epoch=len(train_labels)//FLAGS.batch_size, epochs=1, verbose=1)    
        model.evaluate(dev_dataset, callbacks=[fmetric])
        model.save(FLAGS.save_folder_path+"/{0}/model{1}.h5".format(folder_name, str(i+1)))


if __name__ == "__main__":

    absl.flags.mark_flag_as_required('model_folder_path')
    absl.flags.mark_flag_as_required('experiment_index')
    absl.app.run(main)