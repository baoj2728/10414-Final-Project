import sys
sys.path.append('./python')
import needle as ndl
import needle.nn as nn
import math
import numpy as np
np.random.seed(0)

class ResNet9(ndl.nn.Module):
    def __init__(self, device=None, dtype="float32"):
        super().__init__()
        ### BEGIN YOUR SOLUTION ###
        def ConvBN(in_channels, out_channels, kernel_size, stride):
            return ndl.nn.Sequential(
                ndl.nn.Conv(in_channels, out_channels, kernel_size, stride,
                            device=device, dtype=dtype),
                ndl.nn.BatchNorm2d(out_channels, device=device, dtype=dtype),
                ndl.nn.ReLU()
            )

        self.model = ndl.nn.Sequential(
            ConvBN(3, 16, 7, 4),
            ConvBN(16, 32, 3, 2),
            ndl.nn.Residual(ndl.nn.Sequential(
                ConvBN(32, 32, 3, 1),
                ConvBN(32, 32, 3, 1)
            )),
            ConvBN(32, 64, 3, 2),
            ConvBN(64, 128, 3, 2),
            ndl.nn.Residual(ndl.nn.Sequential(
                ConvBN(128, 128, 3, 1),
                ConvBN(128, 128, 3, 1)
            )),
            ndl.nn.Flatten(),
            ndl.nn.Linear(128, 128, device=device, dtype=dtype),
            ndl.nn.ReLU(),
            ndl.nn.Linear(128, 10, device=device, dtype=dtype)
        )
        ### END YOUR SOLUTION

    def forward(self, x):
        ### BEGIN YOUR SOLUTION
        return self.model(x)
        ### END YOUR SOLUTION


class LanguageModel(nn.Module):
    def __init__(self, embedding_size, output_size, hidden_size, num_layers=1,
                 seq_model='rnn', seq_len=40, device=None, dtype="float32", flash=False):
        super(LanguageModel, self).__init__()
        ### BEGIN YOUR SOLUTION
        self.embedding = nn.Embedding(output_size, embedding_size, device=device, dtype=dtype)
        self.seq_model = seq_model
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        if seq_model == "rnn":
            self.rnn = nn.RNN(embedding_size, hidden_size, num_layers=num_layers, device=device, dtype=dtype)
        elif seq_model == "lstm":
            self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers=num_layers, device=device, dtype=dtype)
        elif seq_model == "transformer":
            self.rnn = nn.Transformer(embedding_size, hidden_size=hidden_size, num_layers=num_layers, device=device, dtype=dtype, flash=flash)
        else:
            raise ValueError("seq_model must be 'rnn' or 'lstm'")
        self.linear = nn.Linear(embedding_size if seq_model == "transformer" else hidden_size, output_size, device=device, dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, x, h=None):
        ### BEGIN YOUR SOLUTION
        x = self.embedding(x)
        out, h = self.rnn(x, h)
        seq_len, bs, _ = out.shape
        out = out.reshape((seq_len * bs, out.shape[-1]))
        out = self.linear(out)
        return out, h
        ### END YOUR SOLUTION


if __name__ == "__main__":
    model = ResNet9()
    x = ndl.ops.randu((1, 32, 32, 3), requires_grad=True)
    model(x)
    cifar10_train_dataset = ndl.data.CIFAR10Dataset("data/cifar-10-batches-py", train=True)
    train_loader = ndl.data.DataLoader(cifar10_train_dataset, 128, ndl.cpu(), dtype="float32")
    print(cifar10_train_dataset[1][0].shape)
