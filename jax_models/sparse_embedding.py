from __future__ import annotations
import equinox as eqx
import jax
import jax.numpy as jnp
from typing import NamedTuple

class SparseEmbedding(eqx.Module):
    weights: jax.Array  # [N, D]
    def __init__(self, num_embeddings: int, embedding_dim: int, *, key: jax.Array):
        self.weights = jax.random.normal(key, (num_embeddings, embedding_dim)) * 0.0
    def __call__(self, idx: jax.Array, dtype=jnp.bfloat16):
        return jnp.take(self.weights, idx, axis=0).astype(dtype)

class SparseEmbState(NamedTuple):
    pass

def sparse_sign_sgd_update(embedding: SparseEmbedding, grads: jax.Array, idx: jax.Array, *, lr: float, weight_decay: float):
    # grads: [B, D]; idx: [B]
    # Aggregate duplicate ids
    unique_ids, inv = jnp.unique(idx, return_inverse=True)
    accum = jnp.zeros((unique_ids.shape[0], grads.shape[-1]), dtype=grads.dtype)
    accum = accum.at[inv].add(grads)
    w = embedding.weights
    slices = w[unique_ids]
    # sign sgd with decoupled weight decay
    slices = slices * (1 - lr * weight_decay) - lr * jnp.sign(accum)
    w = w.at[unique_ids].set(slices)
    return eqx.tree_at(lambda e: e.weights, embedding, w)
