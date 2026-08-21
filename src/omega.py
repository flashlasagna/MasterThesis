"""
omega.py — import alias for 08_omega.py
Python cannot import files whose names start with a digit.
Place this file in the same src/ folder as 08_omega.py.
"""
import importlib.util
import pathlib
import sys

_src_dir = pathlib.Path(__file__).parent
_target  = _src_dir / '08_omega.py'

_spec = importlib.util.spec_from_file_location('omega', _target)
_mod  = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

# Re-export everything from the numbered module
sys.modules[__name__].__dict__.update(
    {k: v for k, v in _mod.__dict__.items() if not k.startswith('__')}
)
