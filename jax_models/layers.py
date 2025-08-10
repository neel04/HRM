import math
from typing import Tuple

import jax
import jax.numpy as jnp
import equinox as eqx

Array = jax.Array
CosSin = Tuple[Array, Array]

def trunc_normal(key: Array, shape, std: float = 1.0, lower: float = -2.0, upper: float = 2.0, dtype=jnp.float32):
    # Sample from normal then reject via inverse error function (approx like torch impl)
    # Follow jax reference: sample uniform in [erf(a/sqrt2), erf(b/sqrt2)], apply erfinv
    sqrt2 = math.sqrt(2.0)
    a = math.erf(lower / sqrt2)
    b = math.erf(upper / sqrt2)
    u = jax.random.uniform(key, shape, dtype=dtype, minval=a, maxval=b)
    x = jax.scipy.special.erfinv(u)
    x = x * sqrt2 * std
    x = jnp.clip(x, lower * std, upper * std)
    return x

class CastedLinear(eqx.Module):
    weight: Array
    bias: Array | None

    def __init__(self, in_features: int, out_features: int, *, use_bias: bool, key: Array):
        wkey, bkey = jax.random.split(key)
        self.weight = trunc_normal(wkey, (out_features, in_features), std=1.0 / math.sqrt(in_features))
        if use_bias:
            self.bias = jnp.zeros((out_features,), dtype=jnp.float32)
        else:
            self.bias = None

    def __call__(self, x: Array, dtype=jnp.bfloat16) -> Array:
        w = self.weight.astype(dtype)
        y = jnp.einsum('...i,oi->...o', x, w)
        if self.bias is not None:
            y = y + self.bias.astype(dtype)
        return y

class CastedEmbedding(eqx.Module):
    weight: Array

    def __init__(self, num_embeddings: int, embedding_dim: int, *, init_std: float, key: Array):
        self.weight = trunc_normal(key, (num_embeddings, embedding_dim), std=init_std)

    def __call__(self, idx: Array, dtype=jnp.bfloat16) -> Array:
        return jnp.take(self.weight.astype(dtype), idx, axis=0)

class RotaryEmbedding(eqx.Module):
    cos_cached: Array
    sin_cached: Array

    def __init__(self, dim: int, max_position_embeddings: int, base: float, *, key: Array | None = None):
        inv_freq = 1.0 / (base ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / dim))
        t = jnp.arange(max_position_embeddings, dtype=jnp.float32)
        freqs = jnp.einsum('i,j->ij', t, inv_freq)
        emb = jnp.concatenate([freqs, freqs], axis=-1)
        self.cos_cached = jnp.cos(emb)
        self.sin_cached = jnp.sin(emb)

    def __call__(self) -> CosSin:
        return self.cos_cached, self.sin_cached

def rotate_half(x: Array):
    x1, x2 = jnp.split(x, 2, axis=-1)
    return jnp.concatenate([-x2, x1], axis=-1)

def apply_rotary_pos_emb(q: Array, k: Array, cos: Array, sin: Array):
    # q,k: [B, T, H, D]
    # cos,sin: [T, D]
    q_embed = (q * cos[:, None, :]) + (rotate_half(q) * sin[:, None, :])
    k_embed = (k * cos[:, None, :]) + (rotate_half(k) * sin[:, None, :])
    return q_embed, k_embed

class Attention(eqx.Module):
    hidden_size: int
    head_dim: int
    num_heads: int
    num_key_value_heads: int
    causal: bool

    qkv_proj: CastedLinear
    o_proj: CastedLinear

    def __init__(self, hidden_size: int, head_dim: int, num_heads: int, num_key_value_heads: int, causal: bool, *, key: Array):
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.causal = causal
        k1, k2 = jax.random.split(key)
        self.qkv_proj = CastedLinear(hidden_size, (num_heads + 2 * num_key_value_heads) * head_dim, use_bias=False, key=k1)
        self.o_proj = CastedLinear(num_heads * head_dim, hidden_size, use_bias=False, key=k2)

    def __call__(self, hidden_states: Array, cos_sin: CosSin | None, *, dtype=jnp.bfloat16) -> Array:
        B, T, _ = hidden_states.shape
        qkv = self.qkv_proj(hidden_states, dtype=dtype)
        qkv = qkv.reshape(B, T, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)
        q = qkv[:, :, : self.num_heads]
        k = qkv[:, :, self.num_heads: self.num_heads + self.num_key_value_heads]
        v = qkv[:, :, self.num_heads + self.num_key_value_heads:]
        if cos_sin is not None:
            cos, sin = cos_sin
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
        # Expand k,v for multiquery if needed
        if self.num_key_value_heads == 1 and self.num_heads > 1:
            k = jnp.broadcast_to(k, (B, T, self.num_heads, self.head_dim))
            v = jnp.broadcast_to(v, (B, T, self.num_heads, self.head_dim))
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_scores = jnp.einsum('bthd,bThd->bhtT', q, k) * scale
        if self.causal:
            mask = jnp.tril(jnp.ones((T, T), dtype=bool))
            attn_scores = jnp.where(mask[None, None, :, :], attn_scores, -1e9)
        attn = jax.nn.softmax(attn_scores, axis=-1)
        context = jnp.einsum('bhtT,bThd->bthd', attn, v)
        context = context.reshape(B, T, self.num_heads * self.head_dim)
        return self.o_proj(context, dtype=dtype)

class SwiGLU(eqx.Module):
    gate_up: CastedLinear
    down: CastedLinear

    def __init__(self, hidden_size: int, expansion: float, *, key: Array):
        inter = math.ceil(expansion * hidden_size * 2 / 3 / 256) * 256
        k1, k2 = jax.random.split(key)
        self.gate_up = CastedLinear(hidden_size, inter * 2, use_bias=False, key=k1)
        self.down = CastedLinear(inter, hidden_size, use_bias=False, key=k2)

    def __call__(self, x: Array, dtype=jnp.bfloat16) -> Array:
        gu = self.gate_up(x, dtype=dtype)
        gate, up = jnp.split(gu, 2, axis=-1)
        return self.down(jax.nn.silu(gate) * up, dtype=dtype)

def rms_norm(x: Array, eps: float):
    x_fp32 = x.astype(jnp.float32)
    var = jnp.mean(x_fp32 * x_fp32, axis=-1, keepdims=True)
    x_norm = x_fp32 * jax.lax.rsqrt(var + eps)
    return x_norm.astype(x.dtype)
