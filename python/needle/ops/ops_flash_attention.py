"""FlashAttention Implementation"""

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



class FlashAttentionForward(TensorTupleOp):
    def __init__(self, causal=False):
        self.causal = causal

    def compute(self, q, k, v):
        return q.flash_attention(k, v, self.causal)

    def gradient(self, out_grad, node):
        q, k, v = node.inputs
        output = node[0]
        lse = node[1]

        grads = FlashAttentionBackward(self.causal)(
            q, k, v, output, lse, out_grad[0]
        )

        return grads[0], grads[1], grads[2]

class FlashAttentionBackward(TensorTupleOp):
    def __init__(self, causal=False):
        self.causal = causal

    def compute(self, q, k, v, output, lse, dout):
        return q.flash_attention_backward(
            k, v, output, lse, dout, self.causal
        )

    def gradient(self, out_grad, node):
        raise RuntimeError(
            "Higher-order FlashAttention gradients "
            "are not supported"
        )


def flash_attention(q, k, v, causal=False):
    results = FlashAttentionForward(causal)(q, k, v)
    return results[0]