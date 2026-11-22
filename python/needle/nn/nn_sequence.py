"""The module.
"""
from typing import List
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
from .nn_basic import Parameter, Module



class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.power_scalar(1 + ops.exp(-x), -1)
        ### END YOUR SOLUTION


class RNNCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.nonlinearity = nonlinearity
        assert nonlinearity in ("tanh", "relu")
        bound = 1 / np.sqrt(hidden_size)
        self.W_ih = Parameter(init.rand(input_size, hidden_size, low=-bound, high=bound, device=device, dtype=dtype))
        self.W_hh = Parameter(init.rand(hidden_size, hidden_size, low=-bound, high=bound, device=device, dtype=dtype))
        self.bias_ih = Parameter(init.rand(hidden_size, low=-bound, high=bound, device=device, dtype=dtype)) if bias else None
        self.bias_hh = Parameter(init.rand(hidden_size, low=-bound, high=bound, device=device, dtype=dtype)) if bias else None
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        ### BEGIN YOUR SOLUTION
        if h is None:
            h = init.zeros(X.shape[0], self.hidden_size, device=X.device, dtype=X.dtype)
        out = X @ self.W_ih + h @ self.W_hh
        if self.bias_ih is not None:
            out = out + ops.broadcast_to(ops.reshape(self.bias_ih, (1, self.hidden_size)), out.shape)
            out = out + ops.broadcast_to(ops.reshape(self.bias_hh, (1, self.hidden_size)), out.shape)
        return ops.tanh(out) if self.nonlinearity == "tanh" else ops.relu(out)
        ### END YOUR SOLUTION


class RNN(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn_cells = [
            RNNCell(input_size if i == 0 else hidden_size, hidden_size, bias, nonlinearity, device, dtype)
            for i in range(num_layers)
        ]
        ### END YOUR SOLUTION

    def forward(self, X, h0=None):
        ### BEGIN YOUR SOLUTION
        seq_len, bs, _ = X.shape
        if h0 is None:
            h0 = init.zeros(self.num_layers, bs, self.hidden_size, device=X.device, dtype=X.dtype)
        states = list(ops.split(h0, axis=0))
        inputs = list(ops.split(X, axis=0))
        final_states = []
        for i, cell in enumerate(self.rnn_cells):
            h = states[i]
            outputs = []
            for x in inputs:
                h = cell(x, h)
                outputs.append(h)
            inputs = outputs
            final_states.append(h)
        return ops.stack(inputs, axis=0), ops.stack(final_states, axis=0)
        ### END YOUR SOLUTION


class LSTMCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, device=None, dtype="float32"):
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.input_size = input_size
        self.hidden_size = hidden_size
        bound = 1 / np.sqrt(hidden_size)
        self.W_ih = Parameter(init.rand(input_size, 4 * hidden_size, low=-bound, high=bound, device=device, dtype=dtype))
        self.W_hh = Parameter(init.rand(hidden_size, 4 * hidden_size, low=-bound, high=bound, device=device, dtype=dtype))
        self.bias_ih = Parameter(init.rand(4 * hidden_size, low=-bound, high=bound, device=device, dtype=dtype)) if bias else None
        self.bias_hh = Parameter(init.rand(4 * hidden_size, low=-bound, high=bound, device=device, dtype=dtype)) if bias else None
        self.sigmoid = Sigmoid()
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        ### BEGIN YOUR SOLUTION
        bs = X.shape[0]
        if h is None:
            h0 = init.zeros(bs, self.hidden_size, device=X.device, dtype=X.dtype)
            c0 = init.zeros(bs, self.hidden_size, device=X.device, dtype=X.dtype)
        else:
            h0, c0 = h
        gates = X @ self.W_ih + h0 @ self.W_hh
        if self.bias_ih is not None:
            gates = gates + ops.broadcast_to(ops.reshape(self.bias_ih, (1, 4 * self.hidden_size)), gates.shape)
            gates = gates + ops.broadcast_to(ops.reshape(self.bias_hh, (1, 4 * self.hidden_size)), gates.shape)
        gates = ops.reshape(gates, (bs, 4, self.hidden_size))
        i, f, g, o = ops.split(gates, axis=1)
        i = self.sigmoid(i)
        f = self.sigmoid(f)
        g = ops.tanh(g)
        o = self.sigmoid(o)
        c = f * c0 + i * g
        h = o * ops.tanh(c)
        return h, c
        ### END YOUR SOLUTION


class LSTM(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm_cells = [
            LSTMCell(input_size if i == 0 else hidden_size, hidden_size, bias, device, dtype)
            for i in range(num_layers)
        ]
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        ### BEGIN YOUR SOLUTION
        seq_len, bs, _ = X.shape
        if h is None:
            h0 = init.zeros(self.num_layers, bs, self.hidden_size, device=X.device, dtype=X.dtype)
            c0 = init.zeros(self.num_layers, bs, self.hidden_size, device=X.device, dtype=X.dtype)
        else:
            h0, c0 = h
        h_states = list(ops.split(h0, axis=0))
        c_states = list(ops.split(c0, axis=0))
        inputs = list(ops.split(X, axis=0))
        final_h, final_c = [], []
        for layer, cell in enumerate(self.lstm_cells):
            ht, ct = h_states[layer], c_states[layer]
            outputs = []
            for x in inputs:
                ht, ct = cell(x, (ht, ct))
                outputs.append(ht)
            inputs = outputs
            final_h.append(ht)
            final_c.append(ct)
        return ops.stack(inputs, axis=0), (ops.stack(final_h, axis=0), ops.stack(final_c, axis=0))
        ### END YOUR SOLUTION


class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype="float32"):
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = Parameter(init.randn(num_embeddings, embedding_dim, device=device, dtype=dtype))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        seq_len, bs = x.shape
        one_hot = init.one_hot(self.num_embeddings, x, device=x.device, dtype=self.weight.dtype)
        one_hot = ops.reshape(one_hot, (seq_len * bs, self.num_embeddings))
        out = one_hot @ self.weight
        return ops.reshape(out, (seq_len, bs, self.embedding_dim))
        ### END YOUR SOLUTION
