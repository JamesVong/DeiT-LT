# Re-export everything from the original losses.py so that
# `from losses import DistillationLoss` keeps working now that
# losses/ is a package (which shadows losses.py at the module level).
import importlib.util
import pathlib

_spec = importlib.util.spec_from_file_location(
    "_losses_original",
    pathlib.Path(__file__).parent.parent / "losses.py",
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

DistillationLoss         = _mod.DistillationLoss
DistillationLossMultiCrop = _mod.DistillationLossMultiCrop
