#!/usr/bin/python3

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.layers import Conv1D
from tensorflow.keras.layers import SimpleRNNCell, RNN, Input, Reshape, Layer
from tensorflow.keras.layers import GlobalMaxPooling1D
import numpy as np
import csv

INPUT_WDITH = 9
INPUT_LENGTH = 20
STATE_SIZE = 1
PSC_LENGTH = 5

class SNNLayer(Layer):
    @staticmethod
    def InvokeRNN(cell, inputs):
        state = [cell.get_initial_state(inputs, None, None)]
        for i in range(inputs.shape[1]):
            output, state = cell(inputs[:,i,:], state)
            n_output = tf.sigmoid(output)
            yield (output, n_output)
            #refraction period
            state[0] = tf.math.multiply(state[0], 1-n_output)

    def __init__(self, units, psc_length, **kwargs):
        super().__init__(self, **kwargs)
        self.units = units
        self.psc_length = psc_length

    def build(self, input_shape):
        def add_layer(layer):
            [self._trainable_weights.append(x) for x in layer.trainable_weights]
        if not len(input_shape) == 3:
            raise ValueError("expected ndim=3")
        self.INPUT_LENGTH = input_shape[1]
        self.INPUT_WIDTH = input_shape[2]
        with tf.variable_scope(self.name):
            # PSC integration as Conv1D, has only 1 channel (=>PSC is uniform), shared among all synapses
            self.psc = Conv1D(1, self.psc_length, name='psc', data_format='channels_last', trainable=True)
            self.psc.build((None, input_shape[1], 1))
            # add psc weights to self
            add_layer(self.psc)
        # RNN unit, has only 1 unit (one neuron)
        self.rnn = SimpleRNNCell(self.units, activation=None)
        self.rnn.build((None, 1, self.INPUT_WIDTH))
        add_layer(self.rnn)

    def call(self, inputs, **kwargs):
        # The same PSC is applied to all inputs channels
        syn_inputs = tf.concat([self.psc(x[:,:,i:i+1]) for i in range(self.INPUT_WIDTH)], axis=-1)
        # then the RNN units are called
        o = tf.concat([o for _, o in SNNLayer.InvokeRNN(self.rnn, syn_inputs)], axis=-1)
        return o
# two toy patterns
with open('pattern1.csv','r') as f:
    r=csv.reader(f)
    pattern1 = [row for row in r]

with open('pattern2.csv','r') as f:
    r=csv.reader(f)
    pattern2 = [row for row in r]

x_train = np.stack([pattern1, pattern2])
# assign labels
labels = np.array([1, 0])

x = Input((INPUT_LENGTH, INPUT_WDITH,))
z = SNNLayer(1, 5)(x)
z = Reshape((z.shape[1], 1))(z)
z = GlobalMaxPooling1D()(z)

model = Model(x, z)
model.summary()
model.compile(loss='binary_crossentropy',
    optimizer='rmsprop',
    metrics=['acc'])

model.fit(x_train, labels, batch_size=2, epochs=500, verbose=1, validation_data=(x_train, labels))