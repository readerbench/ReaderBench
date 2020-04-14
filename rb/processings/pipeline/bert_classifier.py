from typing import Dict, List

import numpy as np
import tensorflow as tf
from rb.core.lang import Lang
from rb.processings.encoders.bert import BertWrapper
from rb.processings.pipeline.dataset import Dataset, Task
from rb.processings.pipeline.estimator import Classifier
from tensorflow import keras


class BertClassifier(Classifier):

    def __init__(self, dataset: Dataset, tasks: List[Task], params: Dict[str, object]):
        super().__init__(dataset, tasks, params)
        self.max_seq_length = 128
        self.bert = BertWrapper(dataset.train_docs[0].lang, max_seq_len=self.max_seq_length)
        self.tasks = tasks
        self.model = self.create_model()
        self.bert.load_weights()
        
    def create_model(self) -> keras.Model:
        inputs, bert_output = self.bert.create_inputs_and_model()
        cls_output = self.bert.get_output(bert_output, "cls")
        features = tf.keras.layers.Input(shape=(len(self.dataset.features),), dtype=tf.float32, name="features")
        outputs = []
        global_hidden = tf.keras.layers.Dense(128)
        for i, task in enumerate(self.tasks):
            masked_features = keras.layers.Lambda(lambda x: x * task.mask)(features)
            concat = keras.layers.concatenate([cls_output, masked_features])
            hidden = global_hidden(cls_output)
            output = keras.layers.Dense(len(task.classes), activation='softmax')(hidden)
            outputs.append(output)
        model = keras.Model(inputs=inputs + [features], outputs=outputs)
        optimizer = keras.optimizers.Adam(learning_rate=1e-6)
        model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model



    def train(self):
        train_features = [[doc.indices[feature] for feature in self.dataset.features]
                          for doc in self.dataset.train_docs]
        train_inputs = self.bert.process_input(doc.text for doc in self.dataset.train_docs)
        train_inputs.append(np.array(train_features))
        train_outputs = [np.array(task.get_train_targets()) for task in self.tasks]
        dev_features = [[doc.indices[feature] for feature in self.dataset.features]
                          for doc in self.dataset.dev_docs]
        dev_inputs = self.bert.process_input(doc.text for doc in self.dataset.dev_docs)
        dev_inputs.append(np.array(dev_features))
        dev_outputs = [np.array(task.get_dev_targets()) for task in self.tasks]
        
        self.model.fit(train_inputs, train_outputs, batch_size=32, epochs=10, validation_data=[dev_inputs, dev_outputs])
