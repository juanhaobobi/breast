# Copyright 2017 Abien Fred Agarap
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================

"""Implementation of the Multilayer Perceptron using TensorFlow"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__version__ = "0.1.0"
__author__ = "Abien Fred Agarap"

import numpy as np
import os
import sys
import time
import tensorflow as tf


import numpy as np
import os
import time
import tensorflow as tf

class MLP:
    def __init__(self, alpha, batch_size, node_size, num_classes, num_features):
        self.alpha = alpha
        self.batch_size = batch_size
        self.node_size = node_size
        self.num_classes = num_classes
        self.num_features = num_features

        def build_graph():
            input_shape = (None, self.num_features)
            self.model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=input_shape),
                tf.keras.layers.Dense(self.node_size[0], activation='relu', name='h1_layer'),
                tf.keras.layers.Dense(self.node_size[1], activation='relu', name='h2_layer'),
                tf.keras.layers.Dense(self.node_size[2], activation='relu', name='h3_layer'),
                tf.keras.layers.Dense(self.num_classes, activation='softmax', name='output_layer')
            ])

            self.model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=self.alpha),
                               loss=tf.keras.losses.CategoricalCrossentropy(),
                               metrics=['accuracy'])

        print("\n<log> Building Graph...")
        build_graph()
        print("</log>\n")

    def train(self, num_epochs, train_data, train_size, test_data, test_size, log_dir, result_path):
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        train_data_batched = (train_data[0][:train_size], train_data[1][:train_size])
        test_data_batched = (test_data[0][:test_size], test_data[1][:test_size])

        self.model.fit(train_data_batched[0], tf.keras.utils.to_categorical(train_data_batched[1], self.num_classes),
                       validation_data=(test_data_batched[0], tf.keras.utils.to_categorical(test_data_batched[1], self.num_classes)),
                       epochs=num_epochs, batch_size=self.batch_size, callbacks=[tensorboard_callback])

        predictions = self.model.predict(test_data_batched[0])
        self.save_labels(predictions=predictions, actual=test_data_batched[1], result_path=result_path, phase="testing", step=0)

    @staticmethod
    def save_labels(predictions, actual, result_path, phase, step):
        if not os.path.exists(path=result_path):
            os.mkdir(result_path)

        labels = np.concatenate((predictions, actual.reshape(-1, 1)), axis=1)
        np.save(file=os.path.join(result_path, "{}-mlp-{}.npy".format(phase, step)), arr=labels)
