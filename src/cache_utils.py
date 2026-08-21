"""
cache_utils.py — helpers for the results cache written by main.py and
read by rerun_outputs.py.

The numbered source modules (02_dma_dms.py, 03_ml_models.py, ...) are
loaded through alias modules, so their result dataclasses cannot be
re-imported by pickle. We therefore store results as plain nested dicts
and rebuild attribute-style objects (SimpleNamespace) on load. All
downstream code accesses results by attribute (res.forecasts, res.msfe,
dma_result.r2_oos[...]) and never by isinstance, so the substitution is
transparent.
"""
import types
import numpy as np
import pandas as pd

_LEAF = (np.ndarray, pd.DataFrame, pd.Series, pd.Index, str, bytes, int, float,
         bool, type(None))

_MARK = '__result_obj__'


def to_plain(obj):
    """Recursively convert dataclass/namespace results to plain dicts."""
    if isinstance(obj, _LEAF):
        return obj
    if hasattr(obj, '__dataclass_fields__') or isinstance(obj, types.SimpleNamespace):
        d = {k: to_plain(v) for k, v in vars(obj).items()}
        d[_MARK] = True
        return d
    if isinstance(obj, dict):
        return {k: to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(to_plain(v) for v in obj)
    return obj


def from_plain(obj):
    """Inverse of to_plain: dicts tagged as results become SimpleNamespace."""
    if isinstance(obj, dict):
        if obj.get(_MARK):
            return types.SimpleNamespace(
                **{k: from_plain(v) for k, v in obj.items() if k != _MARK})
        return {k: from_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(from_plain(v) for v in obj)
    return obj