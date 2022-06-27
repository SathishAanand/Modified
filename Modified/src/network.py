import os

import tensorflow as tf

from tensorflow.keras import Input, backend
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import TimeDistributed, Dense, GRU
from tensorflow.keras.layers.experimental.preprocessing import Normalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.callbacks import ModelCheckpoint

#from src import NetworkConfiguration, Datasplits
from trainer import *
from dataloader import *
from configuration import *


class TurbulanceNetwork:
    def __init__(self, config: NetworkConfiguration):

        self.config = config

        self.model_name = self.config.model_name
        self.model_id = self.config.model_id
        self.model_extension = f'{self.model_name}_{self.model_id}.h5'

        # Build Layers
        self.model = Sequential()
        self.model.add(Normalization())
        self.model.add(Input(shape=(self.config.sequence_lenght, self.config.batch_size)))
        self.model.add(self._build_time_distributed_layer(self.config.hidden_dimensions[0]))
        self.model.add(self._build_time_distributed_layer(self.config.hidden_dimensions[1]))
        self.model.add(GRU(self.config.hidden_dimensions[1], activation='tanh', return_sequences=False, stateful=False))
        self.model.add(self._build_dense_layer(self.config.hidden_dimensions[2]))
        self.model.add(self._build_dense_layer(self.config.hidden_dimensions[3]))
        self.model.add(self._build_dense_layer(self.config.batch_size))

        # Detect GPU for accelerating the computation
        gpus = tf.config.experimental.list_physical_devices('GPU')

        if len(gpus) != 0:
            self.device = '/gpu:0'
        else:
            self.device = '/cpu:0'

    @staticmethod
    def _build_time_distributed_layer(hidden_dimension: int) -> TimeDistributed:
        return TimeDistributed(Dense(hidden_dimension,
                                     activation='relu',
                                     kernel_initializer='he_uniform'))

    @staticmethod
    def _build_dense_layer(hidden_dimension: int) -> Dense:
        return Dense(hidden_dimension, activation='relu', kernel_initializer='he_uniform')

    def correlation_coefficient_metric(self, y_true, y_pred):
        x = y_true
        y = y_pred
        mx = backend.mean(x)
        my = backend.mean(y)
        xm, ym = x-mx, y-my
        r_num = backend.sum(tf.multiply(xm, ym))
        xm_sq = backend.sum(backend.square(xm))
        ym_sq = backend.sum(backend.square(ym))
        r_den = backend.sqrt(tf.multiply(xm_sq, ym_sq))
        r = r_num / r_den
        return r

    def train_model(self, data: Datasplits):

        decay_steps = data.training_data.shape[0] / (self.config.batch_size * self.config.decay_epochs)

        lr_schedule = ExponentialDecay(self.config.learning_rate,
                                       decay_steps=decay_steps,
                                       decay_rate=self.config.decay_rate,
                                       staircase=self.config.decay_stair_case)

        # compile model
        self.model.compile(optimizer=Adam(learning_rate=lr_schedule),
                           loss='mse',
                           metrics=self.correlation_coefficient_metric)

        checkpoint = ModelCheckpoint(self.model_extension,
                                     monitor='val_categorical_accuracy',
                                     mode='max',
                                     save_best_only=True,
                                     verbose=1)

        tf.debugging.set_log_device_placement(True)
        with tf.device(self.device):
            print(f'Training on the device: {self.device}\n')

            history = self.model.fit(data.training_data,
                                     data.training_labels,
                                     epochs=self.config.batch_size,
                                     batch_size=self.config.batch_size,
                                     validation_data=(data.testing_data, data.testing_labels),
                                     callbacks=[checkpoint])
        return history

    def load_model(self):
        self.model.load_weights(self.model_extension)
