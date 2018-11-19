#!/usr/bin/python3

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.layers import Conv1D
from tensorflow.keras.layers import SimpleRNNCell, RNN, Input, Reshape, Layer
from tensorflow.keras.layers import GlobalMaxPooling1D
from tensorflow.keras import backend as K
import numpy as np

INPUT_WIDTH = 100
INPUT_LENGTH = 100
NO_OF_PATTERNS = 500
STATE_SIZE = 1
PSC_LENGTH = 10

def gen_pattern(n, length, spking_prob=0.03, seed=1):
    rng = np.random.RandomState(seed)
    return np.array([rng.binomial(1, spking_prob, n) for _ in range(length)])

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
            self.psc_weights = self.psc.trainable_weights
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
# generated patterns
g_rng = np.random.RandomState(seed=3000)
x_train = np.stack([gen_pattern(INPUT_WIDTH, INPUT_LENGTH,seed=i) for i in g_rng.randint(1e6, size=500)])
print(x_train.shape)
# assign labels randomly
labels = g_rng.binomial(1, p=0.1, size=500)

S = tf.Session()
K.set_session(S)

x = Input((x_train.shape[1], x_train.shape[2]))
q = SNNLayer(STATE_SIZE, PSC_LENGTH)
z = q(x)
z = Reshape((z.shape[1], 1))(z)
z = GlobalMaxPooling1D()(z)

model = Model(x, z)
model.summary()
model.compile(loss='binary_crossentropy',
    optimizer='rmsprop',
    metrics=['acc'])

model.fit(x_train, labels, batch_size=50, epochs=500, verbose=1, validation_data=(x_train, labels))