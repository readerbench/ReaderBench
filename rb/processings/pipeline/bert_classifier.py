from typing import Dict, List

import numpy as np
import tensorflow as tf
from rb.core.lang import Lang
from rb.processings.encoders.bert import BertWrapper
from rb.processings.pipeline.dataset import Dataset, TargetType, Task
from rb.processings.pipeline.estimator import Classifier, Regressor
from tensorflow import keras
from sklearn.model_selection import KFold, StratifiedKFold

class BertClassifier(Classifier, Regressor):

    def __init__(self, dataset: Dataset, tasks: List[Task], params: Dict[str, object]):
        super(Classifier, self).__init__(dataset, tasks, params)
        self.max_seq_length = 128
        self.bert = BertWrapper(dataset.lang, max_seq_len=self.max_seq_length)
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
            if task.type is TargetType.FLOAT:
                output = keras.layers.Dense(1, name=f"output{i}")(hidden)
            else:
                output = keras.layers.Dense(len(task.classes), activation='softmax', name=f"output{i}")(hidden)
            outputs.append(output)
        model = keras.Model(inputs=inputs + [features], outputs=outputs)
        optimizer = keras.optimizers.Adam(learning_rate=1e-5)
        losses = {
            f"output{i}": tf.keras.losses.MeanSquaredError() if task.type is TargetType.FLOAT else tf.keras.losses.SparseCategoricalCrossentropy()
            for i, task in enumerate(self.tasks)
        }
        metrics = {
            f"output{i}": tf.keras.metrics.RootMeanSquaredError() if task.type is TargetType.FLOAT else "accuracy"
            for i, task in enumerate(self.tasks)
        }
        model.compile(optimizer=optimizer, loss=losses, metrics=metrics)
        return model

    def cross_validation(self, n=5):
        kf = KFold(n, shuffle=True)
        features = [[indices[feature] for feature in self.dataset.features]
                    for indices in self.dataset.train_features]
        inputs = self.bert.process_input(self.dataset.train_texts)
        inputs.append(np.array(features))
        outputs = [np.array(task.get_train_targets()) for task in self.tasks]
        for train_index, dev_index in kf.split(inputs[0]):
            train_inputs = [input[train_index] for input in inputs]
            dev_inputs = [input[dev_index] for input in inputs]
            train_outputs = [output[train_index] for output in outputs]
            dev_outputs = [output[dev_index] for output in outputs]
            history = self.model.fit(train_inputs, train_outputs, batch_size=32, epochs=10, validation_data=[dev_inputs, dev_outputs])
            print(history.history.keys())
            
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
