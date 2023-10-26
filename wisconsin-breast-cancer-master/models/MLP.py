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

"""Implementation of the Multilayer Perceptron using TensorFlow 2.x"""
import numpy as np
import os
import sys
import time
import tensorflow as tf

class MLP:
    """Implementation of the Multilayer Perceptron using TensorFlow 2.x"""

    def __init__(self, alpha, batch_size, node_size, num_classes, num_features):
        """Initialize the MLP model

        Parameters
        ----------
        alpha : float
          The learning rate to be used by the neural network.
        batch_size : int
          The number of batches to use for training/validation/testing.
        node_size : list of int
          The number of neurons in each hidden layer of the neural network.
        num_classes : int
          The number of classes in a dataset.
        num_features : int
          The number of features in a dataset.
        """
        self.alpha = alpha
        self.batch_size = batch_size
        self.node_size = node_size
        self.num_classes = num_classes
        self.num_features = num_features

        self.optimizer = tf.keras.optimizers.SGD(learning_rate=self.alpha)
        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
        self.train_loss_metric = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

        self.test_loss_metric = tf.keras.metrics.Mean(name='test_loss')
        self.test_accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

        self.build_model()

    def build_model(self):
        """Build the MLP model architecture"""
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(self.node_size[0], activation='relu', input_shape=(self.num_features,)),
            tf.keras.layers.Dense(self.node_size[1], activation='relu'),
            tf.keras.layers.Dense(self.node_size[2], activation='relu'),
            tf.keras.layers.Dense(self.num_classes, activation='softmax')
        ])

    def train_step(self, data):
        """Training step function"""
        x, y = data
        with tf.GradientTape() as tape:
            predictions = self.model(x)
            loss = self.loss_fn(y, predictions)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        self.train_loss_metric(loss)
        self.train_accuracy_metric(y, predictions)

    def test_step(self, data):
        """Test step function"""
        x, y = data
        predictions = self.model(x, training=False)
        t_loss = self.loss_fn(y, predictions)
        self.test_loss_metric(t_loss)
        self.test_accuracy_metric(y, predictions)

    def train(self, num_epochs, train_dataset, test_dataset):
        """Trains the MLP model

        Parameters
        ----------
        num_epochs : int
          The number of passes over the entire dataset.
        train_dataset : tf.data.Dataset
          The training dataset.
        test_dataset : tf.data.Dataset
          The testing dataset.
        """
        for epoch in range(num_epochs):
            for batch in train_dataset:
                self.train_step(batch)

            for test_batch in test_dataset:
                self.test_step(test_batch)

            template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
            print(template.format(epoch + 1,
                                  self.train_loss_metric.result(),
                                  self.train_accuracy_metric.result() * 100,
                                  self.test_loss_metric.result(),
                                  self.test_accuracy_metric.result() * 100))
    
            # Reset the metrics for the next epoch
            self.train_loss_metric.reset_states()
            self.train_accuracy_metric.reset_states()
            self.test_loss_metric.reset_states()
            self.test_accuracy_metric.reset_states()

    def save_labels(self, predictions, actual, result_path, phase, step):
        """Saves the actual and predicted labels to a NPY file

        Parameters
        ----------
        predictions : numpy.ndarray
          The NumPy array containing the predicted labels.
        actual : numpy.ndarray
          The NumPy array containing the actual labels.
        result_path : str
          The path where to save the concatenated actual and predicted labels.
        phase : str
          The phase for which the predictions is, i.e. training/validation/testing.
        step : int
          The time step for the NumPy arrays.
        """
        if not os.path.exists(path=result_path):
            os.mkdir(result_path)
        # Concatenate the predicted and actual labels
        labels = np.concatenate((predictions, actual), axis=1)
        # save every labels array to NPY file
        np.save(
            file=os.path.join(result_path, "{}-mlp-{}.npy".format(phase, step)),
            arr=labels,
        )
