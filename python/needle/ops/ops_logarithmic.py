from typing import Optional, Any, Union
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

from ..backend_selection import array_api, BACKEND 

class LogSoftmax(TensorOp):
    def compute(self, Z: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        m = array_api.max(Z, axis=1, keepdims=True)
        result = array_api.log(array_api.sum(array_api.exp(Z - m), axis=1))
        lse = result + m.reshape(result.shape)
        return Z - lse.reshape((Z.shape[0], 1))
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        Z = node.inputs[0]

        grad_sum = summation(out_grad, axes=(1,))

        grad_sum = reshape(grad_sum, (Z.shape[0], 1))

        grad_sum = broadcast_to(grad_sum, Z.shape)

        return out_grad - exp(node) * grad_sum
        ### END YOUR SOLUTION


def logsoftmax(a: Tensor) -> Tensor:
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None) -> None:
        self.axes = axes

    def compute(self, Z: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        if self.axes is None:
            axes = tuple(range(len(Z.shape)))
        elif isinstance(self.axes, int):
            axes = (self.axes,)
        else:
            axes = self.axes

        axes = tuple(sorted(set(ax % len(Z.shape) for ax in axes)))

        shape = list(Z.shape)
        for axis in axes:
            shape[axis] = 1

        shape = tuple(shape)

        m = Z
        for axis in sorted(axes, reverse=True):
            m = m.max(axis=axis)

        m_broadcast = m.reshape(shape).broadcast_to(Z.shape)

        exp_z = (Z - m_broadcast).exp()

        result = exp_z
        for axis in sorted(axes, reverse=True):
            result = result.sum(axis=axis)

        return result.log() + m
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        Z = node.inputs[0]

        if self.axes is None:
            axes = tuple(range(len(Z.shape)))
        elif isinstance(self.axes, int):
            axes = (self.axes,)
        else:
            axes = self.axes

        shape = list(Z.shape)
        for axis in axes:
            shape[axis] = 1
        shape = tuple(shape)

        lse = broadcast_to(reshape(node, shape), Z.shape)
        grad = broadcast_to(reshape(out_grad, shape), Z.shape)

        return grad * exp(Z - lse)
        ### END YOUR SOLUTION


def logsumexp(a: Tensor, axes: Optional[tuple] = None) -> Tensor:
    return LogSumExp(axes=axes)(a)