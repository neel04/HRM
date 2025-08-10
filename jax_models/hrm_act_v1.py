from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Sequence
import math

import jax
import jax.numpy as jnp
import equinox as eqx

from .layers import (
    CastedEmbedding, CastedLinear, RotaryEmbedding, Attention, SwiGLU, rms_norm, CosSin
)
from .sparse_embedding import SparseEmbedding

Array = jax.Array

class InnerCarry(eqx.Module):
    z_H: Array
    z_L: Array

class OuterCarry(eqx.Module):
    inner: InnerCarry
    steps: Array  # int32 [B]
    halted: Array  # bool [B]
    current_data: Dict[str, Array]

class Config(eqx.Module):
    batch_size: int
    seq_len: int
    puzzle_emb_ndim: int
    num_puzzle_identifiers: int
    vocab_size: int
    H_cycles: int
    L_cycles: int
    H_layers: int
    L_layers: int
    hidden_size: int
    expansion: float
    num_heads: int
    pos_encodings: str
    rms_norm_eps: float
    rope_theta: float
    halt_max_steps: int
    halt_exploration_prob: float
    forward_dtype: str

class Block(eqx.Module):
    attn: Attention
    mlp: SwiGLU
    norm_eps: float
    def __init__(self, cfg: Config, *, key: Array):
        k1, k2 = jax.random.split(key)
        self.attn = Attention(cfg.hidden_size, cfg.hidden_size // cfg.num_heads, cfg.num_heads, cfg.num_heads, causal=False, key=k1)
        self.mlp = SwiGLU(cfg.hidden_size, cfg.expansion, key=k2)
        self.norm_eps = cfg.rms_norm_eps
    def __call__(self, hidden_states: Array, *, cos_sin: CosSin | None, dtype=jnp.bfloat16) -> Array:
        h = hidden_states + self.attn(hidden_states, cos_sin, dtype=dtype)
        h = rms_norm(h, self.norm_eps)
        h = h + self.mlp(h, dtype=dtype)
        h = rms_norm(h, self.norm_eps)
        return h

class ReasoningModule(eqx.Module):
    layers: list[Block]
    def __init__(self, cfg: Config, n: int, *, key: Array):
        keys = jax.random.split(key, n)
        self.layers = [Block(cfg, key=k) for k in keys]
    def __call__(self, hidden_states: Array, input_injection: Array, *, cos_sin: CosSin | None, dtype=jnp.bfloat16) -> Array:
        h = hidden_states + input_injection
        for layer in self.layers:
            h = layer(h, cos_sin=cos_sin, dtype=dtype)
        return h

class Inner(eqx.Module):
    cfg: Config
    embed_tokens: CastedEmbedding
    lm_head: CastedLinear
    q_head: CastedLinear
    puzzle_emb: CastedEmbedding | None
    rotary: RotaryEmbedding | None
    embed_pos: CastedEmbedding | None
    H_level: ReasoningModule
    L_level: ReasoningModule
    H_init: Array
    L_init: Array

    def __init__(self, cfg: Config, *, key: Array):
        self.cfg = cfg
        k_seq = jax.random.split(key, 20)
        self.embed_tokens = CastedEmbedding(cfg.vocab_size, cfg.hidden_size, init_std=1.0 / math.sqrt(cfg.hidden_size), key=k_seq[0])
        self.lm_head = CastedLinear(cfg.hidden_size, cfg.vocab_size, use_bias=False, key=k_seq[1])
        self.q_head = CastedLinear(cfg.hidden_size, 2, use_bias=True, key=k_seq[2])
        self.puzzle_emb = None
        self.embed_pos = None
        if cfg.puzzle_emb_ndim > 0:
            # Use SparseEmbedding (sign-SGD updated) initialized to zeros
            self.puzzle_emb = SparseEmbedding(cfg.num_puzzle_identifiers, cfg.puzzle_emb_ndim, key=k_seq[3])
        self.rotary = None
        if cfg.pos_encodings == 'rope':
            self.rotary = RotaryEmbedding(cfg.hidden_size // cfg.num_heads, cfg.seq_len + self.puzzle_emb_len, cfg.rope_theta)
        elif cfg.pos_encodings == 'learned':
            self.embed_pos = CastedEmbedding(cfg.seq_len + self.puzzle_emb_len, cfg.hidden_size, init_std=1.0 / math.sqrt(cfg.hidden_size), key=k_seq[4])
        self.H_level = ReasoningModule(cfg, cfg.H_layers, key=k_seq[5])
        self.L_level = ReasoningModule(cfg, cfg.L_layers, key=k_seq[6])
        # initial states
        self.H_init = jax.random.normal(k_seq[7], (cfg.hidden_size,), dtype=jnp.float32)
        self.L_init = jax.random.normal(k_seq[8], (cfg.hidden_size,), dtype=jnp.float32)

    @property
    def puzzle_emb_len(self):
        return -(self.cfg.puzzle_emb_ndim // -self.cfg.hidden_size) if self.cfg.puzzle_emb_ndim > 0 else 0

    def _input_embeddings(self, inputs: Array, puzzle_identifiers: Array, *, dtype=jnp.bfloat16) -> Array:
        emb = self.embed_tokens(inputs.astype(jnp.int32), dtype=dtype)
        if self.cfg.puzzle_emb_ndim > 0 and self.puzzle_emb is not None:
            p_emb = self.puzzle_emb(puzzle_identifiers, dtype=dtype)
            pad_count = self.puzzle_emb_len * self.cfg.hidden_size - p_emb.shape[-1]
            if pad_count > 0:
                p_emb = jnp.pad(p_emb, ((0,0),(0,pad_count)))
            p_emb = p_emb.reshape(p_emb.shape[0], self.puzzle_emb_len, self.cfg.hidden_size)
            emb = jnp.concatenate([p_emb, emb], axis=1)
        if self.cfg.pos_encodings == 'learned' and self.embed_pos is not None:
            emb = 0.707106781 * (emb + self.embed_pos.weight.astype(dtype))
        return math.sqrt(self.cfg.hidden_size) * emb

    def empty_carry(self, batch_size: int, *, dtype=jnp.bfloat16) -> InnerCarry:
        shape = (batch_size, self.cfg.seq_len + self.puzzle_emb_len, self.cfg.hidden_size)
        z = jnp.empty(shape, dtype=dtype)
        return InnerCarry(z_H=z, z_L=z)

    def reset_carry(self, reset_flag: Array, carry: InnerCarry) -> InnerCarry:
        # reset_flag: [B] bool
        H_init = jnp.broadcast_to(self.H_init, carry.z_H.shape[-1])
        L_init = jnp.broadcast_to(self.L_init, carry.z_L.shape[-1])
        z_H = jnp.where(reset_flag[:, None, None], H_init[None,None,:], carry.z_H)
        z_L = jnp.where(reset_flag[:, None, None], L_init[None,None,:], carry.z_L)
        return InnerCarry(z_H=z_H, z_L=z_L)

    def __call__(self, carry: InnerCarry, batch: Dict[str, Array], *, key: Array, dtype=jnp.bfloat16):
        cos_sin = self.rotary() if self.rotary is not None else None
        inp = self._input_embeddings(batch['inputs'], batch['puzzle_identifiers'], dtype=dtype)
        z_H, z_L = carry.z_H, carry.z_L
        # no grad unrolled loops: use stop_gradient
        def H_body(h_state, _):
            z_H_i, z_L_i = h_state
            def L_loop(z_L_inner):
                z_L_inner = self.L_level(z_L_inner, z_H_i + inp, cos_sin=cos_sin, dtype=dtype)
                return z_L_inner
            # L cycles minus last
            for _l in range(self.cfg.L_cycles - 1):
                z_L_i = jax.lax.stop_gradient(L_loop(z_L_i))
            # Last L cycle with grad in outer section below
            return (z_H_i, z_L_i), None
        for _h in range(self.cfg.H_cycles - 1):
            (z_H, z_L), _ = H_body((z_H, z_L), None)
            z_H = jax.lax.stop_gradient(self.H_level(z_H, z_L, cos_sin=cos_sin, dtype=dtype))
        # Final cycle with grad
        for _l in range(self.cfg.L_cycles):
            z_L = self.L_level(z_L, z_H + inp, cos_sin=cos_sin, dtype=dtype)
        z_H = self.H_level(z_H, z_L, cos_sin=cos_sin, dtype=dtype)
        logits = self.lm_head(z_H, dtype=dtype)[:, self.puzzle_emb_len:]
        q_logits = self.q_head(z_H[:,0], dtype=jnp.float32)
        new_carry = InnerCarry(z_H=jax.lax.stop_gradient(z_H), z_L=jax.lax.stop_gradient(z_L))
        return new_carry, logits, (q_logits[...,0], q_logits[...,1])

class ACTModel(eqx.Module):
    cfg: Config
    inner: Inner
    def __init__(self, cfg_dict: dict, *, key: Array):
        cfg = Config(**cfg_dict)
        self.cfg = cfg
        self.inner = Inner(cfg, key=key)
    def initial_carry(self, batch: Dict[str, Array]) -> OuterCarry:
        B = batch['inputs'].shape[0]
        inner_carry = self.inner.empty_carry(B)
        steps = jnp.zeros((B,), dtype=jnp.int32)
        halted = jnp.ones((B,), dtype=bool)
        current_data = {k: jnp.zeros_like(v) for k,v in batch.items()}
        return OuterCarry(inner=inner_carry, steps=steps, halted=halted, current_data=current_data)
    def __call__(self, carry: OuterCarry, batch: Dict[str, Array], *, key: Array):
        inner_carry = self.inner.reset_carry(carry.halted, carry.inner)
        steps = jnp.where(carry.halted, 0, carry.steps)
        def update_current(k, v):
            mask = carry.halted.reshape((-1,) + (1,) * (v.ndim - 1))
            return jnp.where(mask, batch[k], v)
        current_data = {k: update_current(k,v) for k,v in carry.current_data.items()}
        inner_carry, logits, (q_halt_logits, q_continue_logits) = self.inner(inner_carry, current_data, key=key)
        outputs = dict(logits=logits, q_halt_logits=q_halt_logits, q_continue_logits=q_continue_logits)
        steps = steps + 1
        is_last = steps >= self.cfg.halt_max_steps
        halted = is_last
        # exploration + halting only if training (we pass a flag via cfg.forward_dtype hack? better param)
        # For now skip exploration; add later.
        # target q
        def compute_target(inner_carry2: InnerCarry):
            _, _, (nqh, nqc) = self.inner(inner_carry2, current_data, key=key)
            return jax.nn.sigmoid(jnp.where(is_last, nqh, jnp.maximum(nqh, nqc)))
        target_q_continue = compute_target(inner_carry)
        outputs['target_q_continue'] = target_q_continue
        new_carry = OuterCarry(inner=inner_carry, steps=steps, halted=halted, current_data=current_data)
        return new_carry, outputs
