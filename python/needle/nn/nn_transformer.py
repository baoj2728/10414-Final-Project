from typing import List
from needle.autograd import Tensor
import needle.backend_ndarray.ndarray as ndarray
from needle import ops
import needle.init as init
import numpy as np
from .nn_sequence import Embedding
from .nn_basic import (
    Parameter, 
    Module, 
    ReLU,
    Dropout,
    LayerNorm1d,
    Linear,
    Sequential
)


class MultiHeadAttention(Module):
    """
    The multi-head self attention module.
    """
    def __init__(
        self,
        *,
        dropout = 0.,
        causal = False,
        device = None,
        dtype = "float32",
    ):

        super().__init__()

        self.device = device
        self.dtype = dtype

        self.causal = causal
        self.dropout = Dropout(dropout)

    def create_causal_mask(self, i, j, device):
        """
        return a triangular causal mask.
        Input: i, j: the shape of the mask to be created
        """
        mask = -np.finfo(np.float32).max * np.triu(
            np.ones((1, 1, i, j), dtype=np.float32), j - i + 1)

        return ndarray.array(
            mask, device=device)

    def matmul(self, a, b_transpose):
        """
        batched matrix multiplication;
        """
        batch_shape = a.shape[:-2]
        m, k = a.shape[-2:]
        n = b_transpose.shape[-2]

        a = a.reshape((*batch_shape, m, 1, k))
        b_transpose = b_transpose.reshape((*batch_shape, 1, n, k))

        shape = (*batch_shape, m, n, k)
        a = a.broadcast_to(shape)
        b_transpose = b_transpose.broadcast_to(shape)

        return (a * b_transpose).sum(axes=len(shape) - 1)

    def softmax(self, logit):
        """
        The softmax function; 
        """
        max_val = Tensor(
            logit.realize_cached_data().max(axis=3),
            device=logit.device,
            dtype=logit.dtype,
            requires_grad=False
        )

        max_val = max_val.reshape((*logit.shape[:-1], 1))
        max_val = max_val.broadcast_to(logit.shape)

        probs = ops.exp(logit - max_val)

        denom = probs.sum(axes=3)
        denom = denom.reshape((*logit.shape[:-1], 1))
        denom = denom.broadcast_to(logit.shape)

        return probs / denom

    def forward(
        self,
        q, k, v,
    ):
        """
        The forward function of the MultiHeadAttention activation function.
        Input: three states q, k, v, with shape (batch_size, num_head, seq_len, dim_head)
        Output: the activation output `result` and attention softmax probability `probs` (with dropout applied)
        """
        batch_size, num_head, queries_len, q_dim = q.shape
        _, _, keys_values_len, k_dim = k.shape
        _, _, _, v_dim = v.shape

        assert q_dim == k_dim == v_dim

        logits = self.matmul(q, k) / np.sqrt(q_dim)

        if self.causal:
            mask = Tensor(
                self.create_causal_mask(queries_len, keys_values_len, q.device),
                device=q.device, dtype=q.dtype, requires_grad=False
            )
            logits = logits + mask.broadcast_to(logits.shape)

        probs = self.dropout(self.softmax(logits))
        result = self.matmul(probs, ops.transpose(v, axes=(2, 3)))

        return result, probs

class MultiHeadFlashAttention(Module):
    """
    Multi-head self-attention using FlashAttention.
    """

    def __init__(
        self,
        *,
        dropout=0.,
        causal=False,
        device=None,
        dtype="float32",
    ):
        super().__init__()

        self.device = device
        self.dtype = dtype

        self.causal = causal
        self.dropout = Dropout(dropout)

        if dropout != 0:
            raise ValueError(
                "FlashAttention dropout is not supported yet"
            )

    def forward(self, q, k, v):
        """
        Input:
            q: (batch_size, num_head, queries_len, dim_head)
            k: (batch_size, num_head, keys_values_len, dim_head)
            v: (batch_size, num_head, keys_values_len, dim_head)

        Output:
            result: (batch_size, num_head, queries_len, dim_head)
        """

        batch_size, num_head, queries_len, q_dim = q.shape
        _, _, keys_values_len, k_dim = k.shape
        _, _, _, v_dim = v.shape

        assert q_dim == k_dim == v_dim

        result = ops.flash_attention(
            q, k, v, causal=self.causal
        )

        return result


class AttentionLayer(Module):

    def __init__(
        self,
        q_features: int,
        num_head: int,
        dim_head: int,
        *,
        k_features: int = None,
        v_features: int = None,
        out_features: int = None,
        dropout = 0.,
        causal = True,
        device = None,
        dtype = "float32",
        flash = False,
    ):

        super().__init__()

        self.device = device
        self.dtype = dtype
        self.flash = flash

        if k_features is None:
            k_features = q_features
        if v_features is None:
            v_features = q_features
        if out_features is None:
            out_features = q_features

        self.q_features = q_features
        self.k_features = k_features
        self.v_features = v_features
        self.out_features = out_features

        self.num_head = num_head
        self.dim_head = dim_head

        self.prenorm_q = LayerNorm1d(
            q_features, device=device, dtype=dtype)
        self.prenorm_k = LayerNorm1d(
            k_features, device=device, dtype=dtype)
        self.prenorm_v = LayerNorm1d(
            v_features, device=device, dtype=dtype)

        inner_dim = num_head * dim_head
        
        self.q_projection = Linear(
            q_features, inner_dim, bias=False,
            device=device, dtype=dtype)
        self.k_projection = Linear(
            k_features, inner_dim, bias=False,
            device=device, dtype=dtype)
        self.v_projection = Linear(
            v_features, inner_dim, bias=False,
            device=device, dtype=dtype)

        if not flash:
            self.attn = MultiHeadAttention(
                dropout=dropout, causal=causal,
                device=device, dtype=dtype)
        else:
            self.attn = MultiHeadFlashAttention(
                dropout=dropout, causal=causal,
                device=device, dtype=dtype)
            
        self.out_projection = Linear(
            inner_dim, out_features, bias=False,
            device=device, dtype=dtype)

    def forward(
        self,
        q, k=None, v=None,
    ):
        """
        The forward function of the self-attention layer.
        Input: `q` with shape (batch_size, q_len, q_dim)
               `k` (if not None) with shape (batch_size, kv_len, k_dim)
               `v` (if not None) with shape (batch_size, kv_len, v_dim)
        Output: the output `result` with shape (batch_size, kv_len, out_features)
        """

        if k is None:
            k = q
        if v is None:
            v = q

        batch_size, queries_len, q_dim = q.shape
        _, keys_values_len, k_dim = k.shape
        _, _, v_dim = v.shape

        result = None

        ### BEGIN YOUR SOLUTION
        q = self.prenorm_q(q.reshape((batch_size * queries_len, q_dim)))
        k = self.prenorm_k(k.reshape((batch_size * keys_values_len, k_dim)))
        v = self.prenorm_v(v.reshape((batch_size * keys_values_len, v_dim)))

        q = self.q_projection(q).reshape((batch_size, queries_len, self.num_head, self.dim_head))
        k = self.k_projection(k).reshape((batch_size, keys_values_len, self.num_head, self.dim_head))
        v = self.v_projection(v).reshape((batch_size, keys_values_len, self.num_head, self.dim_head))

        q = ops.transpose(q, axes=(1, 2))
        k = ops.transpose(k, axes=(1, 2))
        v = ops.transpose(v, axes=(1, 2))

        if self.flash:
            result = self.attn(q, k, v)
        else:
            result, _ = self.attn(q, k, v)
        result = ops.transpose(result, axes=(1, 2))
        result = result.reshape((batch_size * queries_len, self.num_head * self.dim_head))
        result = self.out_projection(result)
        result = result.reshape((batch_size, queries_len, self.out_features))
        ### END YOUR SOLUTION

        return result

class TransformerLayer(Module):

    def __init__(
        self,
        q_features: int,
        num_head: int,
        dim_head: int,
        hidden_size: int,
        *,
        dropout = 0.,
        causal = True,
        device = None,
        dtype = "float32",
        flash = False
    ):

        super().__init__()

        self.device = device
        self.dtype = dtype
        self.flash = flash

        ### BEGIN YOUR SOLUTION
        self.attn = AttentionLayer(
            q_features, num_head, dim_head, dropout=dropout,
            causal=causal, device=device, dtype=dtype, flash=flash
        )
        self.norm = LayerNorm1d(q_features, device=device, dtype=dtype)
        self.ffn = Sequential(
            Linear(q_features, hidden_size, device=device, dtype=dtype),
            ReLU(),
            Dropout(dropout),
            Linear(hidden_size, q_features, device=device, dtype=dtype)
        )
        self.dropout = Dropout(dropout)
        ### END YOUR SOLUTION

    def forward(
        self,
        x
    ):
        """
        The forward function of a Transformer Layer.
        Input: the hidden states from previous layers `x` with shape (batch_size, seq_len, x_dim)
        Ouput: the hidden states after the Transformer Layer `x` with shape (batch_size, seq_len, x_dim)
        """

        batch_size, seq_len, x_dim = x.shape

        x = x + self.dropout(self.attn(x))

        y = self.norm(x.reshape((batch_size * seq_len, x_dim)))
        y = self.ffn(y)
        y = y.reshape((batch_size, seq_len, x_dim))

        x = x + self.dropout(y)
        return x


class Transformer(Module):

    def __init__(
        self,
        embedding_size: int,
        hidden_size: int,
        num_layers: int, 
        *,
        num_head: int = 8,
        dim_head: int = 64,
        dropout = 0.,
        causal = True,
        device = None,
        dtype = "float32",
        flash = False,
        batch_first = False,
        sequence_len = 2048
    ):

        super().__init__()

        self.device = device
        self.dtype = dtype
        self.batch_first = batch_first

        ### BEGIN YOUR SOLUTION
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.pos_embedding = Embedding(sequence_len, embedding_size, device=device, dtype=dtype)
        self.layers = Sequential(*[
            TransformerLayer(
                embedding_size, num_head, dim_head, hidden_size,
                dropout=dropout, causal=causal, device=device, dtype=dtype, flash=flash
            )
            for _ in range(num_layers)
        ])
        ### END YOUR SOLUTION

    def forward(
        self,
        x, h=None
    ):

        if not self.batch_first:
            x = ops.transpose(x, axes=(0, 1))

        ### BEGIN YOUR SOLUTION
        batch_size, seq_len, _ = x.shape
        positions = np.broadcast_to(
            np.arange(seq_len, dtype=np.float32), (batch_size, seq_len)
        ).copy()
        positions = Tensor(positions, device=x.device, dtype=x.dtype, requires_grad=False)
        x = x + self.pos_embedding(positions)
        x = self.layers(x)
        ### END YOUR SOLUTION

        if not self.batch_first:
            x = ops.transpose(x, axes=(0, 1))

        return x, init.zeros_like(x)