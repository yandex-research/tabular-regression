# For optimizers with conflicting or easy-to-misspell names,
# submodules are imported instead of optimizer classes.
from vendor import c_muon, muon  # noqa: F401
from vendor.adan import Adan  # noqa: F401
from vendor.ademamix import AdEMAMix  # noqa: F401
from vendor.adopt import ADOPT  # noqa: F401
from vendor.lion_pytorch import Lion  # noqa: F401
from vendor.sam import SAM  # noqa: F401
from vendor.soap import SOAP  # noqa: F401

from . import c_adamw, muon_original_precision, muon_tuned_abc, normuon  # noqa: F401
from .adamw_sam import AdamWSharpnessAware  # noqa: F401
from .conda_torch import Conda  # noqa: F401
from .gsam import GSAM  # noqa: F401
from .kron import Kron  # noqa: F401
from .mars_m import MARS_M  # noqa: F401
from .signum import Signum  # noqa: F401
from .splus import SPlus  # noqa: F401
