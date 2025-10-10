from .core.lattice import OscillinkLattice  # noqa: F401
from .core.perf import compare_perf  # noqa: F401
from .core.provenance import compare_provenance  # noqa: F401
from .core.receipts import verify_receipt, verify_receipt_mode  # noqa: F401
from .preprocess.diffusion import compute_diffusion_gates  # noqa: F401

# Public alias preferred in docs
Oscillink = OscillinkLattice

__all__ = [
	"Oscillink",
	"OscillinkLattice",
	"verify_receipt",
	"verify_receipt_mode",
	"compare_perf",
	"compare_provenance",
	"compute_diffusion_gates",
]
__version__ = "0.1.5"
