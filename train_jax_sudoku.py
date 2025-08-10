import argparse, time
import jax, jax.numpy as jnp, optax, equinox as eqx
from typing import Any, Dict
from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig
from jax_models.hrm_act_v1 import ACTModel
from jax_models.losses import ACTLossHead
from jax_models.data import jax_batch_iterator
from jax_models.sparse_embedding import SparseEmbedding, sparse_sign_sgd_update


def build_dataset(path: str, batch_size: int, seed: int):
    cfg = PuzzleDatasetConfig(
        seed=seed,
        dataset_path=path,
        global_batch_size=batch_size,
        test_set_mode=False,
        epochs_per_iter=1,
        rank=0,
        num_replicas=1,
    )
    return PuzzleDataset(cfg, split='train')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_path', type=str, required=True)
    ap.add_argument('--steps', type=int, default=100)
    ap.add_argument('--batch_size', type=int, default=384)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--puzzle_emb_lr', type=float, default=1e-4)
    ap.add_argument('--weight_decay', type=float, default=1.0)
    ap.add_argument('--puzzle_emb_weight_decay', type=float, default=1.0)
    ap.add_argument('--halt_max_steps', type=int, default=3)
    ap.add_argument('--log_every', type=int, default=10)
    ap.add_argument('--eval_every', type=int, default=100)
    ap.add_argument('--eval_batches', type=int, default=5)
    args = ap.parse_args()

    dataset = build_dataset(args.data_path, args.batch_size, seed=0)
    first = next(iter(dataset))
    _, batch0_torch, _ = first
    from jax_models.data import pytorch_batch_to_jax
    batch0 = pytorch_batch_to_jax(batch0_torch)
    seq_len = batch0['inputs'].shape[1]

    cfg = dict(
        batch_size=args.batch_size,
        seq_len=seq_len,
        puzzle_emb_ndim=32,
        num_puzzle_identifiers=int(batch0['puzzle_identifiers'].max())+1,
        vocab_size=int(batch0['inputs'].max())+1,
        H_cycles=2,
        L_cycles=2,
        H_layers=2,
        L_layers=2,
        hidden_size=256,
        expansion=4.0,
        num_heads=8,
        pos_encodings='learned',
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        halt_max_steps=args.halt_max_steps,
        halt_exploration_prob=0.0,
        forward_dtype='bfloat16'
    )
    key = jax.random.PRNGKey(0)
    model = ACTModel(cfg, key=key)
    loss_head = ACTLossHead(model, loss_type='softmax_cross_entropy')

    @eqx.filter_value_and_grad(has_aux=True)
    def loss_fn(loss_head, carry, batch, key):
        carry, loss, metrics, _, _ = loss_head(carry, batch, return_keys=[], key=key)
        return loss, (carry, metrics)

    dense_params = eqx.filter(loss_head, lambda x: eqx.is_array(x) and not isinstance(x, jax.Array) or True)
    optimizer = optax.adamw(args.lr, weight_decay=args.weight_decay)
    opt_state = optimizer.init(eqx.filter(loss_head, eqx.is_array))

    from jax_models.data import prefetch_jax_batches
    iterator = prefetch_jax_batches(iter(dataset), prefetch=2, device_put=True)
    carry = loss_head.initial_carry(batch0)

    # Simple evaluation function
    def evaluate(eval_batches=5):
        it = jax_batch_iterator(iter(dataset))
        total = dict(count=0, accuracy=0.0, exact_accuracy=0, q_halt_accuracy=0, steps=0)
        key_eval = key
        for _ in range(eval_batches):
            try:
                _s, b, effb = next(it)
            except StopIteration:
                break
            c = loss_head.initial_carry(b)
            # iterate ACT until halted or max steps
            for _m in range(cfg['halt_max_steps']):
                key_eval, sk2 = jax.random.split(key_eval)
                c, loss, metrics, _, all_finish = loss_head(c, b, return_keys=[], key=sk2)
                # accumulate
                for k in total:
                    total[k] += float(metrics.get(k, 0.0))
                if all_finish:
                    break
        if total['count'] > 0:
            acc = total['accuracy']/total['count']
            ex = total['exact_accuracy']/total['count']
        else:
            acc = ex = 0.0
        return dict(accuracy=acc, exact_accuracy=ex)

    for step in range(args.steps):
        try:
            _set, batch, eff = next(iterator)
        except StopIteration:
            iterator = prefetch_jax_batches(iter(dataset), prefetch=2, device_put=True)
            _set, batch, eff = next(iterator)
        key, sk = jax.random.split(key)
        (loss, (carry, metrics)), grads = loss_fn(loss_head, carry, batch, sk)
        # Separate sparse embedding
        puzzle_emb = loss_head.model.inner.puzzle_emb
        sparse_grads = None
        if isinstance(puzzle_emb, SparseEmbedding):
            def is_sparse_leaf(x):
                return x is puzzle_emb.weights
            sparse_grads = eqx.filter(grads, is_sparse_leaf)
            dense_grads = eqx.filter(grads, lambda x: (x is not puzzle_emb.weights) and eqx.is_array(x))
        else:
            dense_grads = grads
        params = eqx.filter(loss_head, eqx.is_array)
        updates, opt_state = optimizer.update(dense_grads, opt_state, params)
        loss_head = eqx.apply_updates(loss_head, updates)
        if isinstance(puzzle_emb, SparseEmbedding) and sparse_grads is not None:
            from jax.tree_util import tree_leaves
            leaves = tree_leaves(sparse_grads)
            if leaves:
                gW = leaves[0]
                used = jnp.unique(batch['puzzle_identifiers'])
                g_used = gW[used]
                new_emb = sparse_sign_sgd_update(puzzle_emb, g_used, used, lr=args.puzzle_emb_lr, weight_decay=args.puzzle_emb_weight_decay)
                loss_head = eqx.tree_at(lambda lh: lh.model.inner.puzzle_emb, loss_head, new_emb)
        if step % args.log_every == 0:
            cnt = float(metrics.get('count', 0) or 1)
            acc = float(metrics.get('accuracy', 0)/cnt) if cnt>0 else 0.0
            exacc = float(metrics.get('exact_accuracy', 0)/cnt) if cnt>0 else 0.0
            print(f'step {step} loss {float(loss):.4f} acc {acc:.4f} ex {exacc:.4f} effB {eff}')
        if step>0 and step % args.eval_every == 0:
            eval_metrics = evaluate(args.eval_batches)
            print(f'[eval] step {step} accuracy {eval_metrics["accuracy"]:.4f} exact {eval_metrics["exact_accuracy"]:.4f}')
    final_eval = evaluate(args.eval_batches)
    print(f'Final eval accuracy {final_eval["accuracy"]:.4f} exact {final_eval["exact_accuracy"]:.4f}')
    # sample predictions for first eval batch
    it = jax_batch_iterator(iter(dataset))
    try:
        _s, b, _e = next(it)
        c = loss_head.initial_carry(b)
        key_samp = key
        detached_logits = None
        for _m in range(cfg['halt_max_steps']):
            key_samp, sks = jax.random.split(key_samp)
            c, _loss_tmp, metrics_tmp, detached, all_finish = loss_head(c, b, return_keys=['logits'], key=sks)
            if 'logits' in detached:
                detached_logits = detached['logits']
            if all_finish:
                break
        import numpy as np
        if detached_logits is not None:
            preds = jnp.argmax(detached_logits, axis=-1)[:2]
            print('Sample preds (first 2 examples):')
            print(np.array(preds))
            print('Labels:')
            print(np.array(b['labels'][:2]))
    except StopIteration:
        pass
    print('Done')

if __name__ == '__main__':
    main()
