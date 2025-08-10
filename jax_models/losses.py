from __future__ import annotations
from typing import Sequence, Dict, Any, Tuple
import equinox as eqx
import jax
import jax.numpy as jnp

IGNORE_LABEL_ID = -100

# stablemax analog

def s_fn(x, epsilon=1e-30):
    return jnp.where(x < 0, 1.0 / (1 - x + epsilon), x + 1)

def log_stablemax(x, axis=-1):
    s_x = s_fn(x)
    return jnp.log(s_x / jnp.sum(s_x, axis=axis, keepdims=True))

def stablemax_cross_entropy(logits, labels, ignore_index=-100):
    logprobs = log_stablemax(logits.astype(jnp.float64), axis=-1)
    valid = labels != ignore_index
    transformed = jnp.where(valid, labels, 0)
    pred_lp = jnp.take_along_axis(logprobs, transformed[...,None], axis=-1)[...,0]
    return -jnp.where(valid, pred_lp, 0)

def softmax_cross_entropy(logits, labels, ignore_index=-100):
    # logits: [B, T, V]
    logp = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)
    valid = labels != ignore_index
    transformed = jnp.where(valid, labels, 0)
    gathered = jnp.take_along_axis(logp, transformed[...,None], axis=-1)[...,0]
    return -jnp.where(valid, gathered, 0)

class ACTLossHead(eqx.Module):
    model: Any
    loss_type: str
    def __init__(self, model, loss_type: str):
        self.model = model
        self.loss_type = loss_type
    def initial_carry(self, *args, **kwargs):
        return self.model.initial_carry(*args, **kwargs)
    def __call__(self, carry, batch: Dict[str,jax.Array], return_keys: Sequence[str], *, key: jax.Array):
        new_carry, outputs = self.model(carry=carry, batch=batch, key=key)
        labels = new_carry.current_data['labels']
        loss_fn = globals()[self.loss_type]
        mask = labels != IGNORE_LABEL_ID
        loss_counts = jnp.sum(mask, axis=-1)
        loss_div = jnp.clip(loss_counts, 1)[:,None]
        is_correct = mask & (jnp.argmax(outputs['logits'], axis=-1) == labels)
        seq_correct = jnp.sum(is_correct, axis=-1) == loss_counts
        valid_metrics = new_carry.halted & (loss_counts > 0)
        lm_loss = jnp.sum(loss_fn(outputs['logits'], labels, ignore_index=IGNORE_LABEL_ID) / loss_div)
        q_halt_logits = outputs['q_halt_logits']
        targets = jnp.where(seq_correct, 1.0, 0.0)
        q_halt_loss = jnp.sum(jnp.maximum(q_halt_logits, 0) - q_halt_logits * targets + jnp.log1p(jnp.exp(-jnp.abs(q_halt_logits))))
        metrics = dict(
            count=jnp.sum(valid_metrics),
            accuracy=jnp.sum(jnp.where(valid_metrics, jnp.sum(is_correct.astype(jnp.float32)/loss_div, axis=-1), 0.0)),
            exact_accuracy=jnp.sum((valid_metrics & seq_correct).astype(jnp.int32)),
            q_halt_accuracy=jnp.sum((valid_metrics & ((q_halt_logits>=0)==seq_correct)).astype(jnp.int32)),
            steps=jnp.sum(jnp.where(valid_metrics, new_carry.steps, 0)),
            lm_loss=lm_loss,
            q_halt_loss=q_halt_loss,
        )
        q_continue_loss = 0.0
        if 'target_q_continue' in outputs:
            t2 = outputs['target_q_continue']
            logits2 = outputs['q_continue_logits']
            q_continue_loss = jnp.sum(jnp.maximum(logits2, 0) - logits2 * t2 + jnp.log1p(jnp.exp(-jnp.abs(logits2))))
            metrics['q_continue_loss'] = q_continue_loss
        detached = {k: outputs[k] for k in return_keys if k in outputs}
        total_loss = lm_loss + 0.5*(q_halt_loss + q_continue_loss)
        all_finish = jnp.all(new_carry.halted)
        return new_carry, total_loss, metrics, detached, all_finish
