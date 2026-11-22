"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks

from ..backend_selection import array_api, BACKEND
from .ops_tuple import *


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return a ** b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, rhs = node.inputs
        return out_grad * (rhs * lhs ** (rhs - 1)), out_grad * (lhs ** rhs * log(lhs))
        ### END YOUR SOLUTION


def power(a, b):
    return EWisePow()(a, b)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return a ** self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, = node.inputs
        return (out_grad * (self.scalar * lhs ** (self.scalar - 1)), ) 
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a / b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, rhs = node.inputs
        return (out_grad * 1 / rhs, out_grad * (-lhs) / (rhs ** 2))
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a / self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, = node.inputs
        return (out_grad / self.scalar,)
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        axes = list(range(len(a.shape)))
        i, j = self.axes if self.axes is not None else (-2, -1)
        axes[i], axes[j] = axes[j], axes[i]
        return a.permute(tuple(axes))
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return transpose(out_grad, self.axes)
        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.compact().reshape(self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return reshape(out_grad, node.inputs[0].shape)
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.broadcast_to(a, self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        input_shape = node.inputs[0].shape
        output_shape = self.shape

        axes = []
        offset = len(output_shape) - len(input_shape)
        axes.extend(range(offset))

        for i, dim in enumerate(input_shape):
            if dim == 1 and output_shape[i + offset] != 1:
                axes.append(i + offset)

        grad = summation(out_grad, tuple(axes))

        return reshape(grad, input_shape)
        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        if self.axes is None:
            return a.sum().compact().reshape(())

        axes = (self.axes,) if isinstance(self.axes, int) else self.axes
        result = a
        for axis in sorted((ax % a.ndim for ax in axes), reverse=True):
            result = result.sum(axis=axis)
        return result
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        input_shape = node.inputs[0].shape

        if self.axes is None:
            axes = tuple(range(len(input_shape)))
        elif isinstance(self.axes, int):
            axes = (self.axes,)
        else:
            axes = self.axes

        shape = list(input_shape)
        for axis in axes:
            shape[axis] = 1

        return broadcast_to(
            reshape(out_grad, tuple(shape)),
            input_shape
        )
        ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a @ b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, rhs = node.inputs

        grad_lhs = out_grad @ transpose(rhs)
        grad_rhs = transpose(lhs) @ out_grad

        def sum_to_shape(grad, shape):
            axes = []

            # Extra leading dimensions introduced by broadcasting
            extra = len(grad.shape) - len(shape)
            axes += list(range(extra))

            # Dimensions of size 1 that were broadcast
            for i in range(len(shape)):
                if shape[i] == 1 and grad.shape[i + extra] != 1:
                    axes.append(i + extra)

            if axes:
                grad = summation(grad, axes=tuple(axes))
                grad = reshape(grad, shape)

            return grad

        grad_lhs = sum_to_shape(grad_lhs, lhs.shape)
        grad_rhs = sum_to_shape(grad_rhs, rhs.shape)

        return grad_lhs, grad_rhs
        ### END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return -a
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return (-out_grad, )
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.log(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, = node.inputs
        return (out_grad / lhs, )
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.exp(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, = node.inputs
        return (out_grad * exp(lhs), )
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.maximum(a, 0)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0].realize_cached_data()
        mask = Tensor.make_const(a > 0)
        return out_grad * mask
        ### END YOUR SOLUTION

def relu(a):
    return ReLU()(a)


class Tanh(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.tanh(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad * (1 + -(node * node))
        ### END YOUR SOLUTION


def tanh(a):
    return Tanh()(a)


class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        self.axis = axis

    def compute(self, args: TensorTuple) -> Tensor:
        ### BEGIN YOUR SOLUTION
        shape = list(args[0].shape)
        axis = self.axis % (len(shape) + 1)
        shape.insert(axis, len(args))

        out = array_api.empty(tuple(shape), device=args[0].device)

        for i, a in enumerate(args):
            indices = [slice(None)] * len(shape)
            indices[axis] = slice(i, i + 1)

            expanded_shape = list(a.shape)
            expanded_shape.insert(axis, 1)

            out[tuple(indices)] = a.compact().reshape(
                tuple(expanded_shape)
            )

        return out
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return split(out_grad, self.axis)
        ### END YOUR SOLUTION


def stack(args, axis):
    return Stack(axis)(make_tuple(*args))


class Split(TensorTupleOp):
    def __init__(self, axis: int):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        self.axis = axis

    def compute(self, A):
        ### BEGIN YOUR SOLUTION
        axis = self.axis % len(A.shape)
        shape = list(A.shape)
        n = shape.pop(axis)

        result = []

        for i in range(n):
            indices = [slice(None)] * len(A.shape)
            indices[axis] = slice(i, i + 1)

            a = A[tuple(indices)]
            result.append(a.compact().reshape(tuple(shape)))

        return tuple(result)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return stack(out_grad, self.axis)
        ### END YOUR SOLUTION


def split(a, axis):
    return Split(axis)(a)


class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        axes = self.axes
        if axes is None:
            axes = tuple(range(a.ndim))
        return a.flip(axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return flip(out_grad, self.axes)
        ### END YOUR SOLUTION


def flip(a, axes):
    return Flip(axes)(a)


class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        axes = tuple(axis for axis in self.axes if 0 <= axis < a.ndim)
        shape = list(a.shape)
        for axis in axes:
            shape[axis] *= self.dilation + 1

        out = array_api.empty(tuple(shape), device=a.device)
        out.fill(0)
        slices = tuple(
            slice(None, None, self.dilation + 1) if i in axes
            else slice(None) for i in range(a.ndim)
        )
        out[slices] = a
        return out
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return undilate(out_grad, self.axes, self.dilation)
        ### END YOUR SOLUTION


def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)


class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        axes = tuple(axis for axis in self.axes if 0 <= axis < a.ndim)
        slices = tuple(
            slice(None, None, self.dilation + 1) if i in axes
            else slice(None) for i in range(a.ndim)
        )
        return a[slices].compact()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return dilate(out_grad, self.axes, self.dilation)
        ### END YOUR SOLUTION



def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)
    


class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, A, B):
        ### BEGIN YOUR SOLUTION
        N, H, W, Cin = A.shape
        K, _, _, Cout = B.shape

        if self.padding > 0:
            A = A.pad((
                (0, 0),
                (self.padding, self.padding),
                (self.padding, self.padding),
                (0, 0)
            ))

        H_out = (H + 2 * self.padding - K) // self.stride + 1
        W_out = (W + 2 * self.padding - K) // self.stride + 1

        sN, sH, sW, sC = A.strides

        patches = A.as_strided(
            (N, H_out, W_out, K, K, Cin),
            (sN, sH * self.stride, sW * self.stride, sH, sW, sC)
        )

        patches = patches.compact().reshape(
            (N * H_out * W_out, K * K * Cin)
        )

        kernel = B.compact().reshape((K * K * Cin, Cout))

        out = patches @ kernel

        return out.reshape((N, H_out, W_out, Cout))
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        A, B = node.inputs
        K = B.shape[0]

        if self.stride > 1:
            out_grad = dilate(out_grad, axes=(1, 2), dilation=self.stride - 1)

        flipped_B = flip(B, axes=(0, 1))
        transposed_B = transpose(flipped_B, axes=(2, 3))
        grad_A = conv(out_grad, transposed_B, stride=1, padding=K - 1 - self.padding)

        transposed_A = transpose(A, axes=(0, 3))
        transposed_grad = transpose(out_grad, axes=(0, 1))
        transposed_grad = transpose(transposed_grad, axes=(1, 2))

        grad_B = conv(transposed_A, transposed_grad, stride=1, padding=self.padding)
        grad_B = transpose(grad_B, axes=(0, 1))
        grad_B = transpose(grad_B, axes=(1, 2))

        return grad_A, grad_B
        ### END YOUR SOLUTION


def conv(a, b, stride=1, padding=1):
    return Conv(stride, padding)(a, b)


