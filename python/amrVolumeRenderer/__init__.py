"""Python wrapper around the amrVolumeRenderer Viskores volume renderer."""

from __future__ import annotations

try:
    from amrVolumeRenderer_ext import (  # type: ignore[attr-defined]
        compute_histogram,
        finalize_runtime,
        initialize_runtime,
        render,
    )
except ModuleNotFoundError:  # import packaged extension built into package
    from .amrVolumeRenderer_ext import (  # type: ignore[attr-defined]
        compute_histogram,
        finalize_runtime,
        initialize_runtime,
        render,
    )

__all__ = ["render", "initialize_runtime", "finalize_runtime", "compute_histogram"]
