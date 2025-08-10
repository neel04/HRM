import time
import math
from dataclasses import dataclass
from typing import Dict, Any

import jax
import jax.numpy as jnp
import optax
import equinox as eqx

from jax_models.hrm_act_v1 import ACTModel
from jax_models.losses import ACTLossHead, IGNORE_LABEL_ID, softmax_cross_entropy

@dataclass
class TrainState:
    params: Any
    opt_state: Any
    carry: Any
    step: int

# Minimal config mapping
cfg = dict(
    batch_size=4,
    seq_len=8,
    puzzle_emb_ndim=16,
    num_puzzle_identifiers=10,
    vocab_size=32,
    H_cycles=2,
    L_cycles=2,
    H_layers=2,
    L_layers=2,
    hidden_size=64,
    expansion=4.0,
    num_heads=4,
    pos_encodings="rope",
    rms_norm_eps=1e-5,
    rope_theta=10000.0,
    halt_max_steps=3,
    halt_exploration_prob=0.1,
    forward_dtype="bfloat16",
)

key = jax.random.PRNGKey(0)
model = ACTModel(cfg, key=key)
loss_head = ACTLossHead(model, loss_type='softmax_cross_entropy')

@eqx.filter_value_and_grad(has_aux=True)
def loss_fn(loss_head, carry, batch, key):
    carry, loss, metrics, _, _ = loss_head(carry, batch, return_keys=[], key=key)
    return loss, (carry, metrics)

optimizer = optax.adamw(1e-3)
opt_state = optimizer.init(eqx.filter(loss_head, eqx.is_array))

batch = {
    'inputs': jax.random.randint(key, (cfg['batch_size'], cfg['seq_len']), 0, cfg['vocab_size']),
    'labels': jax.random.randint(key, (cfg['batch_size'], cfg['seq_len']), 0, cfg['vocab_size']),
    'puzzle_identifiers': jax.random.randint(key, (cfg['batch_size'],), 0, cfg['num_puzzle_identifiers']),
}
carry = loss_head.initial_carry(batch)

from jax_models.sparse_embedding import sparse_sign_sgd_update, SparseEmbedding

# Identify sparse embedding path (loss_head.model.inner.puzzle_emb)

for step in range(5):
    key, sk = jax.random.split(key)
    (loss, (carry, metrics)), grads = loss_fn(loss_head, carry, batch, sk)
    # Separate sparse embedding grads
    puzzle_emb = loss_head.model.inner.puzzle_emb if hasattr(loss_head.model.inner, 'puzzle_emb') else None
    if isinstance(puzzle_emb, SparseEmbedding):
        # Extract gradient for weights
        def is_sparse_weight(x):
            return x is puzzle_emb.weights
        sparse_grads = eqx.filter(grads, is_sparse_weight)
        dense_grads = eqx.filter(grads, lambda x: (x is not puzzle_emb.weights) and eqx.is_array(x))
    else:
        dense_grads = grads
        sparse_grads = None
    params = eqx.filter(loss_head, eqx.is_array)
    updates, opt_state = optimizer.update(dense_grads, opt_state, params)
    loss_head = eqx.apply_updates(loss_head, updates)
    # Apply sparse update if available
    if isinstance(puzzle_emb, SparseEmbedding) and sparse_grads is not None:
        # flatten grads to get weights grad; using eqx.tree_leaves on sparse_grads
        leaves = eqx.tree_leaves(sparse_grads)
        if leaves:
            gW = leaves[0]  # full embedding gradient; approximate sign sgd on used indices only by indexing
            # Collect used indices from batch
            used = jnp.unique(batch['puzzle_identifiers'])
            # Slice grads for used
            g_used = gW[used]
            new_emb = sparse_sign_sgd_update(puzzle_emb, g_used, used, lr=1e-3, weight_decay=0.0)
            loss_head = eqx.tree_at(lambda lh: lh.model.inner.puzzle_emb, loss_head, new_emb)
    print('step', step, 'loss', float(loss))

print('JAX minimal training loop finished.')
