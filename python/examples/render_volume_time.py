#!/usr/bin/env python3
"""Example render driver that pins the camera and color scale in Python."""

import math
import os
from pathlib import Path
import sys
import glob


def _configure_import_path() -> None:
    """Allow running the example from a source checkout without installing."""
    repo_root = Path(__file__).resolve().parents[2]
    candidate_paths = [
        repo_root / "python",
        repo_root / "build/lib",
        repo_root / "build/python",
    ]
    for path in candidate_paths:
        if path.exists() and str(path) not in sys.path:
            sys.path.append(str(path))


def _import_renderer():
    try:
        import miniGraphics  # type: ignore[attr-defined]
    except ModuleNotFoundError:
        _configure_import_path()
        import miniGraphics  # type: ignore[attr-defined]
    return miniGraphics


# Hard-coded rendering configuration.
PLOTFILE_GLOB = "../quokka-turb-driving/tests/mach_10_v2/plt*"
VARIABLE = "gasDensity"
IMAGE_WIDTH = 512
IMAGE_HEIGHT = 512
OUTPUT_DIR = Path("renders")
OUTPUT_PREFIX = "render"
ANTIALIASING = 16
BOX_TRANSPARENCY = 0.9
LOG_SCALE = True
CAMERA_EYE = (2.0, 1.2, 2.0)
CAMERA_LOOK_AT = (0.5, 0.5, 0.5)
CAMERA_UP = (0.0, 1.0, 0.0)
FOV_Y = 45.0
NEAR_PLANE = 0.1
FAR_PLANE = 10.0

# Physical scalar -> RGBA ramp used for the volume color map. Values are in the
# original field units; they are mapped through math.log when LOG_SCALE is True.
COLOR_MAP_PHYSICAL = [
    (1.0e-1, 0.04, 0.05, 0.20, 0.00),
    (1.0e0, 0.10, 0.35, 0.70, 0.20),
    (1.0e1, 0.25, 0.70, 0.90, 0.45),
    (1.0e2, 0.65, 0.85, 0.60, 0.65),
    (1.0e3, 0.98, 0.65, 0.30, 0.85),
    (1.0e4, 1.00, 0.98, 0.95, 1.00),
]


def _build_color_map(log_scale: bool):
    transform = math.log if log_scale else (lambda value: value)
    return [
        (
            float(transform(value)),
            float(red),
            float(green),
            float(blue),
            float(alpha),
        )
        for value, red, green, blue, alpha in COLOR_MAP_PHYSICAL
    ]


def _render_frames() -> None:
    renderer = _import_renderer()
    runtime_initialized = False

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    camera_eye = CAMERA_EYE
    camera_look_at = CAMERA_LOOK_AT
    camera_up = CAMERA_UP
    color_map = _build_color_map(LOG_SCALE)
    scalar_range = (
        (color_map[0][0], color_map[-1][0]) if color_map else None
    )

    plotfiles = sorted(glob.glob(PLOTFILE_GLOB))
    NUM_FRAMES = len(plotfiles)
    if NUM_FRAMES <= 0:
        raise ValueError("num-frames must be a positive integer")

    try:
        renderer.initialize_runtime()
        runtime_initialized = True

        for frame_idx in range(NUM_FRAMES):
            output_path = OUTPUT_DIR / f"{OUTPUT_PREFIX}_{frame_idx:04d}.ppm"

            renderer.render(
                plotfile=plotfiles[frame_idx],
                width=IMAGE_WIDTH,
                height=IMAGE_HEIGHT,
                variable=VARIABLE,
                output=str(output_path),
                antialiasing=ANTIALIASING,
                log_scale=LOG_SCALE,
                box_transparency=BOX_TRANSPARENCY,
                scalar_range=scalar_range,
                camera_eye=camera_eye,
                camera_look_at=camera_look_at,
                camera_up=camera_up,
                camera_fov_y=FOV_Y,
                camera_near=NEAR_PLANE,
                camera_far=FAR_PLANE,
                color_map=color_map,
            )
    finally:
        if runtime_initialized:
            renderer.finalize_runtime()


def main() -> None:
    _render_frames()


if __name__ == "__main__":
    main()
