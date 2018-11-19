# tempotron-tensorflow
An incomplete tensorflow based implementation of a CNN-RNN hybrid network modelled after the Tempotron spiking neuron model (https://en.wikipedia.org/wiki/Tempotron). This is a work in progress.

# Reasons for this experiment
Backpropagation is hard on Spiking Neural Network (SNN) since the neurons have non-differenciable activation functions. Most existing researches do not even attempt to do backpropagation. However, given that Backpropagation Through Time (BPTT) already exists in deep learning systems, the question of whether BPTT can be applied to SNNs with a little modification remains. This is a rough answer to that question. We may polish it later to make it more useful.

# Usage
Tested on Python 3.7 and Tensorflow 1.12
Run snn_cell.py to train an SNN network with 100 input neurons on a set of 100x500 input patterns.

# What is going on
The inputs are N sequences of either 0 or 1, which encodes the firing patterns of the input neurons (1=fire, 0=no fire). These go through a Conv1D layer that simulates the effect of Postsynaptic current (PSC). The output of the Conv1D then goes to a SimpleRNNCell that simulates the state change of an integrate-and-fire (IAF) neuron over time. The output of the cell passes through a sigmoid activation function, which simulates activation of the neuron, and finally the state is gated by the output to simulate the effect of a refractory period after each firing. 
Since the goal of Tempotron training is to increase/decrease maximum membrane voltage, a GlobalMaxPooling1D is used to reduce the outputs to 1.

# Difference from the original Tempotron
1) The PSC kernel is trainable in this implementation, not in the original
2) The firing threshold (bias) is trainable in this implementation, not in the original
3) So far there is no way to enforce pre-connection delay (while max delay=length of PSC)

# TODO
1) Multiple IAF neurons
2) Internal connectivity between neurons
