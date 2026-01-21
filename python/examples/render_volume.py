#!/usr/bin/env python3
"""Convenience driver for the amrVolumeRenderer Python bindings.

This mirrors the CLI of the volume_renderer executable but allows the
workflow to be scripted entirely from Python. Run it under ``mpirun`` to drive
multi-rank renders, e.g.:

    mpirun -np 4 python render_volume.py ./plt00010 --width 512 --height 512
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple


def _configure_import_path() -> None:
    """Loosely mimic an editable install when running from the source tree."""
    repo_root = Path(__file__).resolve().parents[2]
    candidate_paths = [
        repo_root / "python",
        repo_root / "build/lib",
        repo_root / "build/python",
    ]
    for path in candidate_paths:
        if path.exists() and str(path) not in sys.path:
            sys.path.append(str(path))


try:
    from amrVolumeRenderer import render  # type: ignore[attr-defined]
except ModuleNotFoundError:
    _configure_import_path()
    from amrVolumeRenderer import render  # type: ignore[attr-defined]


def _parse_up_vector(values: Optional[Sequence[float]]) -> Optional[Tuple[float, float, float]]:
    if values is None:
        return None
    if len(values) != 3:
        raise ValueError("--up-vector expects exactly three floats")
    return float(values[0]), float(values[1]), float(values[2])


def _parse_arguments(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a plotfile with amrVolumeRenderer.")
    parser.add_argument("plotfile", type=Path, help="Path to the AMReX plotfile")
    parser.add_argument("--width", type=int, default=512, help="Image width in pixels (default: 512)")
    parser.add_argument("--height", type=int, default=512, help="Image height in pixels (default: 512)")
    parser.add_argument(
        "--box-transparency",
        type=float,
        default=0.0,
        help="Transparency factor applied per AMR box in [0, 1] (default: 0)",
    )
    parser.add_argument(
        "--antialiasing",
        type=int,
        default=1,
        help="Supersampling factor (must be a perfect square, default: 1)",
    )
    parser.add_argument(
        "--no-visibility-graph",
        dest="visibility_graph",
        action="store_false",
        help="Disable the visibility graph for ordering",
    )
    parser.add_argument(
        "--write-visibility-graph",
        action="store_true",
        help="Export DOT files describing the visibility graph",
    )
    parser.add_argument(
        "--variable",
        type=str,
        default=None,
        help="Scalar variable to render (default: first variable in plotfile)",
    )
    parser.add_argument("--min-level", type=int, default=0, help="Coarsest AMR level to include (default: 0)")
    parser.add_argument(
        "--max-level",
        type=int,
        default=-1,
        help="Finest AMR level to include (-1 renders all available levels, default: -1)",
    )
    parser.add_argument(
        "--log-scale",
        action="store_true",
        help="Apply natural log scaling to positive values before normalization",
    )
    parser.add_argument(
        "--up-vector",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=None,
        help="Custom camera up vector (three floats). Defaults to world +Y.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output image filename (supports .ppm or .png, default: volume-renderer.ppm)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_arguments(argv)

    up_vector = _parse_up_vector(args.up_vector)
    output_path = str(args.output) if args.output is not None else None

    render(
        plotfile=str(args.plotfile),
        width=args.width,
        height=args.height,
        box_transparency=args.box_transparency,
        antialiasing=args.antialiasing,
        visibility_graph=args.visibility_graph,
        write_visibility_graph=args.write_visibility_graph,
        variable=args.variable,
        min_level=args.min_level,
        max_level=args.max_level,
        log_scale=args.log_scale,
        up_vector=up_vector,
        output=output_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
