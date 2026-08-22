"""Name -> baseline constructor. The single place a new baseline is announced.

Imports are lazy: naming a baseline must not drag in every other baseline's
dependencies, and `--list` has to work on a machine with a busy GPU.
"""
from __future__ import annotations

from typing import Callable, Dict

from .common.protocol import SpikeVolumeBaseline

_BUILDERS: Dict[str, Callable[..., SpikeVolumeBaseline]] = {}


def register(name: str):
    def deco(fn):
        if name in _BUILDERS:
            raise KeyError(f"baseline {name!r} already registered")
        _BUILDERS[name] = fn
        return fn
    return deco


def _load_all() -> None:
    from . import dichotomized_gaussian  # noqa: F401
    from . import coupled_glm            # noqa: F401
    from . import maskgit_flat           # noqa: F401
    # Not an external baseline -- the shipped model, wrapped so the diagnostics
    # measure it with the same code. Registered here so `--baseline pipeline`
    # cannot drift onto a different clip set or a different metric path.
    from . import pipeline_reference     # noqa: F401


def available() -> list[str]:
    _load_all()
    return sorted(_BUILDERS)


def build(name: str, **kwargs) -> SpikeVolumeBaseline:
    _load_all()
    if name not in _BUILDERS:
        raise KeyError(f"unknown baseline {name!r}; available: {sorted(_BUILDERS)}")
    return _BUILDERS[name](**kwargs)
