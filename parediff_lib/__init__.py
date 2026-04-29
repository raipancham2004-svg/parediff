"""PareDiff — Pareto-Aware Diffusion for Macro Placement."""
__version__ = "0.1.0"

# ============================================================
# Backward-compatibility shim for checkpoints saved before the
# package was renamed from 'code/' to 'parediff_lib/' on 2026-04-29.
# Old pickles reference 'code.utils.PareDiffConfig' etc. — we make
# those names resolve to the renamed modules.
# ============================================================
import sys as _sys
import types as _types

# Import in dependency order
from . import utils as _utils
from . import gnn_encoder as _gnn_encoder
from . import diffusion_model as _diffusion_model
from . import synthetic_dataset as _synthetic_dataset
try:
    from . import routability_classifier as _routability_classifier
except Exception:
    _routability_classifier = None

_code_mod = _types.ModuleType('code')
_code_mod.utils = _utils
_code_mod.gnn_encoder = _gnn_encoder
_code_mod.diffusion_model = _diffusion_model
_code_mod.synthetic_dataset = _synthetic_dataset
if _routability_classifier is not None:
    _code_mod.routability_classifier = _routability_classifier

_sys.modules['code'] = _code_mod
_sys.modules['code.utils'] = _utils
_sys.modules['code.gnn_encoder'] = _gnn_encoder
_sys.modules['code.diffusion_model'] = _diffusion_model
_sys.modules['code.synthetic_dataset'] = _synthetic_dataset
if _routability_classifier is not None:
    _sys.modules['code.routability_classifier'] = _routability_classifier
