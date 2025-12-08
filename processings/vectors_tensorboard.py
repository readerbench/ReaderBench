#!/usr/bin/env python
# coding: utf8
"""Visualize spaCy word vectors in Tensorboard.

Adapted from: https://gist.github.com/BrikerMan/7bd4e4bd0a00ac9076986148afc06507
"""
from __future__ import unicode_literals

from os import path

import math
import numpy
import spacy
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins.projector import visualize_embeddings, ProjectorConfig
from gensim.models import LsiModel 


def load_model(inputFile: str):
    vocab = []
    matrix = []
    with open(inputFile, "rt") as f:
        n, dim = f.readline().split(" ")
        for line in f.readlines():
            cols = line.split(" ")
            word = cols[0]
            vector = [float(val) for val in cols[1:]]
            vocab.append(word)
            matrix.append(vector)
    return vocab, matrix

if __name__ == "__main__":
    out_loc = "models"
    model_path = "resources/new_config/semantic-models-enea_tasa-en/word2vec.model"
    meta_file = "{}.tsv".format("vocab")
    out_meta_file = path.join(out_loc, meta_file)
    name = "enea-word2vec"
    
    print('Loading vectors model: {}'.format(model_path))
    vocab, matrix = load_model(model_path)

    # model = LsiModel.load("RO/ReadME/lsa.bin")
    # matrix = matrix * numpy.sqrt(model.projection.s)
    
    print('Building Tensorboard Projector metadata for ({}) vectors: {}'.format(len(vocab), meta_file))

    # Store vector data in a tensorflow variable
    
    # Write a tab-separated file that contains information about the vectors for visualization
    #
    # Reference: https://www.tensorflow.org/programmers_guide/embedding#metadata
    with open(out_meta_file, 'wb') as file_metadata:
        # Define columns in the first row
        # file_metadata.write("Text\n".encode('utf-8'))
        # Write out a row for each vector that we add to the tensorflow variable we created
        for word in vocab:
            # https://github.com/tensorflow/tensorflow/issues/9094
            
            # Store vector data and metadata
            file_metadata.write("{}\n".format(word).encode('utf-8'))
            
    print('Running Tensorflow Session...')
    sess = tf.InteractiveSession()
    tf.Variable(matrix, trainable=False, name=name)
    tf.global_variables_initializer().run()
    saver = tf.train.Saver()
    writer = tf.summary.FileWriter(out_loc, sess.graph)

    # Link the embeddings into the config
    config = ProjectorConfig()
    embed = config.embeddings.add()
    embed.tensor_name = name
    embed.metadata_path = meta_file

    # Tell the projector about the configured embeddings and metadata file
    visualize_embeddings(writer, config)

    # Save session and print run command to the output
    print('Saving Tensorboard Session...')
    saver.save(sess, path.join(out_loc, '{}.ckpt'.format(name)))
    print('Done. Run `tensorboard --logdir={0}` to view in Tensorboard'.format(out_loc))

