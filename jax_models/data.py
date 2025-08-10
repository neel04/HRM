import itertools
import numpy as np
import jax
import jax.numpy as jnp
from typing import Iterator, Dict, Any, Iterable, Optional

# Simple adapter that consumes the existing PyTorch PuzzleDataset iterator output
# (set_name, batch_dict, global_effective_batch_size) and yields jax arrays.

def pytorch_batch_to_jax(batch: Dict[str, Any]):
    # batch tensors have .numpy() if on CPU; assume CPU
    out = {}
    for k, v in batch.items():
        np_arr = v.detach().cpu().numpy() if hasattr(v, 'detach') else np.asarray(v)
        out[k] = jnp.array(np_arr)
    return out

def jax_batch_iterator(torch_iter: Iterator, *, device_put: bool = True):
    for set_name, batch, eff in torch_iter:
        jbatch = pytorch_batch_to_jax(batch)
        if device_put:
            jbatch = jax.device_put(jbatch)
        yield set_name, jbatch, eff

class PrefetchIterator:
    def __init__(self, base_iter: Iterable, prefetch: int = 2):
        self.base_iter = iter(base_iter)
        self.prefetch = prefetch
        self.buffer = []
        self._fill()
    def _fill(self):
        try:
            while len(self.buffer) < self.prefetch:
                self.buffer.append(next(self.base_iter))
        except StopIteration:
            pass
    def __iter__(self):
        return self
    def __next__(self):
        if not self.buffer:
            raise StopIteration
        item = self.buffer.pop(0)
        self._fill()
        return item

def prefetch_jax_batches(torch_iter: Iterator, *, prefetch: int = 2, device_put: bool = True):
    return PrefetchIterator(jax_batch_iterator(torch_iter, device_put=device_put), prefetch=prefetch)
