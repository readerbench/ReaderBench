from typing import Dict, List

import numpy as np
import tensorflow as tf
from rb.core.lang import Lang
from rb.processings.encoders.bert import BertWrapper
from rb.processings.pipeline.dataset import Dataset, TargetType, Task
from rb.processings.pipeline.estimator import Classifier, Regressor
from tensorflow import keras
from sklearn.model_selection import KFold, StratifiedKFold
from rb.utils.utils import HiddenPrints

class BertClassifier(Classifier, Regressor):

    def __init__(self, dataset: Dataset, tasks: List[Task], params: Dict[str, object], use_indices=True, use_mask=True, shared=True):
        super(Classifier, self).__init__(dataset, tasks, params)
        self.max_seq_length = 256
        self.bert = BertWrapper(dataset.lang, max_seq_len=self.max_seq_length)
        self.tasks = tasks
        self.output = params["output"]
        self.hidden = params["hidden"]
        self.use_indices = use_indices
        self.use_mask = use_mask
        self.shared = shared
        self.initialize()

    def initialize(self):
        tf.keras.backend.clear_session()
        self.model = self.create_model()
        with HiddenPrints():
            self.bert.load_weights()
        self.initial_weights = self.model.get_weights()

    def create_model(self) -> keras.Model:
        inputs, bert_output = self.bert.create_inputs_and_model()
        cls_output = self.bert.get_output(bert_output, self.output)
        features = tf.keras.layers.Input(shape=(len(self.dataset.features),), dtype=tf.float32, name="features")
        outputs = []
        if self.shared:
            global_hidden = tf.keras.layers.Dense(self.hidden, activation="tanh")
        for i, task in enumerate(self.tasks):
            if self.use_indices:
                if self.use_mask:
                    masked_features = keras.layers.Lambda(lambda x: x * task.mask)(features)
                else:
                    masked_features = features
                concat = keras.layers.concatenate([cls_output, masked_features])
            else:
                concat = cls_output
            if self.shared:
                hidden = global_hidden(concat)
            else:
                hidden = tf.keras.layers.Dense(self.hidden, activation="tanh")(concat)
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
    
    def metric(self, task: Task) -> str:
        return "root_mean_squared_error" if task.type is TargetType.FLOAT else "accuracy"
    
    def cross_validation(self, n=5):
        kf = KFold(n, shuffle=True)
        features = [[indices[feature] for feature in self.dataset.features]
                    for indices in self.dataset.normalized_train_features[:1000]]
        inputs = self.bert.process_input(self.dataset.train_texts[:1000])
        inputs.append(np.array(features))
        outputs = [np.array(task.get_train_targets()[:1000]) for task in self.tasks]
        losses = []
        epochs = []
        for train_index, dev_index in kf.split(inputs[0]):
            train_inputs = [input[train_index] for input in inputs]
            dev_inputs = [input[dev_index] for input in inputs]
            train_outputs = [output[train_index] for output in outputs]
            dev_outputs = [output[dev_index] for output in outputs]
            self.model.set_weights(self.initial_weights)
            history = self.model.fit(train_inputs, train_outputs, batch_size=16, epochs=5, validation_data=[dev_inputs, dev_outputs])
            epoch, loss = min(enumerate(history.history["val_loss"]), key=lambda x: x[1])
            losses.append(loss)
            epochs.append(epoch + 1)
            break
        return int(np.mean(epochs)), np.mean(losses)
            
    def train(self, epochs: int) -> float:
        self.model.set_weights(self.initial_weights)
        train_features = [
            [indices[feature] for feature in self.dataset.features]
            for indices in self.dataset.normalized_train_features[:1000]
        ]
        train_inputs = self.bert.process_input(self.dataset.train_texts[:1000])
        train_inputs.append(np.array(train_features))
        train_outputs = [np.array(task.get_train_targets()[:1000]) for task in self.tasks]
        dev_features = [
            [indices[feature] for feature in self.dataset.features]
            for indices in self.dataset.normalized_dev_features
        ]
        dev_inputs = self.bert.process_input(self.dataset.dev_texts)
        dev_inputs.append(np.array(dev_features))
        dev_outputs = [np.array(task.get_dev_targets()) for task in self.tasks]
        
        history = self.model.fit(train_inputs, train_outputs, batch_size=16, epochs=epochs, validation_data=[dev_inputs, dev_outputs])
        return [history.history[f"val_output{i}_{self.metric(task)}"][-1] for i, task in enumerate(self.tasks)]

    @classmethod
    def parameters(cls):
        return {
            "output": ["cls", "pool"],
            "hidden": [64, 128, 256],
        }
    