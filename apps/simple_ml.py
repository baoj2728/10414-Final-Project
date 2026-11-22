"""hw1/apps/simple_ml.py"""

import struct
import gzip
import numpy as np

import sys

sys.path.append("python/")
import needle as ndl

import needle.nn as nn
from apps.models import *
import time
device = ndl.cpu()

def parse_mnist(image_filename, label_filename):
    ### BEGIN YOUR SOLUTION
    import gzip
    import struct

    with gzip.open(image_filename, "rb") as f:
        _, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        X = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, rows * cols)
        X = X.astype(np.float32)
        X = (X - X.min()) / (X.max() - X.min())

    with gzip.open(label_filename, "rb") as f:
        _, num_labels = struct.unpack(">II", f.read(8))
        y = np.frombuffer(f.read(), dtype=np.uint8).astype(np.int8)

    return X, y
    ### END YOUR SOLUTION


def softmax_loss(Z, y_one_hot):
    ### BEGIN YOUR SOLUTION
    log_sum_exp = ndl.ops.logsumexp(Z, axes=(1,))
    correct_logits = ndl.ops.summation(Z * y_one_hot, axes=(1,))
    return ndl.ops.summation(log_sum_exp - correct_logits) / Z.shape[0]
    ### END YOUR SOLUTION


def nn_epoch(X, y, W1, W2, lr=0.1, batch=100):
    ### BEGIN YOUR SOLUTION
    num_classes = W2.shape[1]

    for i in range(0, X.shape[0], batch):
        X_batch = ndl.Tensor(X[i:i + batch], device=W1.device)
        y_batch = ndl.Tensor(y[i:i + batch], device=W1.device)

        y_one_hot = ndl.init.one_hot(num_classes, y_batch, device=W1.device)

        hidden = ndl.ops.relu(X_batch @ W1)
        logits = hidden @ W2
        loss = softmax_loss(logits, y_one_hot)

        loss.backward()

        W1 = ndl.Tensor(W1.detach() - lr * W1.grad.detach(), device=W1.device, requires_grad=True)
        W2 = ndl.Tensor(W2.detach() - lr * W2.grad.detach(), device=W2.device, requires_grad=True)

    return W1, W2
    ### END YOUR SOLUTION

### CIFAR-10 training ###
def epoch_general_cifar10(dataloader, model, loss_fn=nn.SoftmaxLoss(), opt=None):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    if opt is None:
        model.eval()
    else:
        model.train()

    total_correct = 0
    total_loss = 0
    total_samples = 0

    for X, y in dataloader:
        if opt is not None:
            opt.reset_grad()

        logits = model(X)
        loss = loss_fn(logits, y)

        if opt is not None:
            loss.backward()
            opt.step()

        batch_size = X.shape[0]
        total_correct += np.sum(np.argmax(logits.numpy(), axis=1) == y.numpy())
        total_loss += loss.numpy().item() * batch_size
        total_samples += batch_size

    return total_correct / total_samples, total_loss / total_samples
    ### END YOUR SOLUTION


def train_cifar10(model, dataloader, n_epochs=1, optimizer=ndl.optim.Adam,
                  lr=0.001, weight_decay=0.001, loss_fn=nn.SoftmaxLoss):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss = loss_fn()

    for epoch in range(n_epochs):
        avg_acc, avg_loss = epoch_general_cifar10(
            dataloader, model, loss_fn=loss, opt=opt
        )

    return avg_acc, avg_loss
    ### END YOUR SOLUTION


def evaluate_cifar10(model, dataloader, loss_fn=nn.SoftmaxLoss):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    return epoch_general_cifar10(
        dataloader, model, loss_fn=loss_fn(), opt=None
    )
    ### END YOUR SOLUTION

### PTB training ###

def epoch_general_ptb(data, model, seq_len=40, loss_fn=nn.SoftmaxLoss(), opt=None,
                      clip=None, device=None, dtype="float32"):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    if opt is None:
        model.eval()
    else:
        model.train()

    total_correct = 0
    total_loss = 0
    total_samples = 0
    h = None

    for i in range(0, data.shape[0] - 1, seq_len):
        X, y = ndl.data.get_batch(data, i, seq_len, device=device, dtype=dtype)

        if h is not None:
            if isinstance(h, tuple):
                h = tuple(state.detach() for state in h)
            else:
                h = h.detach()

        if opt is not None:
            opt.reset_grad()

        logits, h = model(X, h)
        loss = loss_fn(logits, y)

        if opt is not None:
            loss.backward()
            if clip is not None:
                opt.clip_grad_norm(clip)
            opt.step()

        batch_size = y.shape[0]
        total_correct += np.sum(np.argmax(logits.numpy(), axis=1) == y.numpy())
        total_loss += loss.numpy().item() * batch_size
        total_samples += batch_size

    return float(total_correct / total_samples), float(total_loss / total_samples)
    ### END YOUR SOLUTION


def train_ptb(model, data, seq_len=40, n_epochs=1, optimizer=ndl.optim.SGD,
              lr=4.0, weight_decay=0.0, loss_fn=nn.SoftmaxLoss, clip=None,
              device=None, dtype="float32"):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss = loss_fn()

    for epoch in range(n_epochs):
        avg_acc, avg_loss = epoch_general_ptb(
            data, model, seq_len=seq_len, loss_fn=loss,
            opt=opt, clip=clip, device=device, dtype=dtype
        )

    return avg_acc, avg_loss
    ### END YOUR SOLUTION


def evaluate_ptb(model, data, seq_len=40, loss_fn=nn.SoftmaxLoss,
                 device=None, dtype="float32"):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    return epoch_general_ptb(
        data, model, seq_len=seq_len, loss_fn=loss_fn(),
        opt=None, device=device, dtype=dtype
    )
    ### END YOUR SOLUTION

### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT


def loss_err(h, y):
    """Helper function to compute both loss and error"""
    y_one_hot = np.zeros((y.shape[0], h.shape[-1]))
    y_one_hot[np.arange(y.size), y] = 1
    y_ = ndl.Tensor(y_one_hot)
    return softmax_loss(h, y_).numpy(), np.mean(h.numpy().argmax(axis=1) != y)
