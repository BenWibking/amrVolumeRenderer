#!/usr/bin/env python3
"""Example render driver that pins the camera and color scale in Python."""

import math
import os
from pathlib import Path
import sys


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
        import amrVolumeRenderer  # type: ignore[attr-defined]
    except ModuleNotFoundError:
        _configure_import_path()
        import amrVolumeRenderer  # type: ignore[attr-defined]
    return amrVolumeRenderer


# Hard-coded rendering configuration.
PLOTFILE = "../quokka-turb-driving/tests/mach_10_v2/plt06550"
VARIABLE = "gasDensity"
IMAGE_WIDTH = 512
IMAGE_HEIGHT = 512
NUM_FRAMES = 36
OUTPUT_DIR = Path("renders")
OUTPUT_PREFIX = "render"
ANTIALIASING = 4
BOX_TRANSPARENCY = 0.9
SCALAR_RANGE = (0.0, 1.0)
LOG_SCALE = True
CAMERA_EYE = (2.0, 1.2, 2.0)
CAMERA_LOOK_AT = (0.5, 0.5, 0.5)
CAMERA_UP = (0.0, 1.0, 0.0)
FOV_Y = 45.0
NEAR_PLANE = 0.1
FAR_PLANE = 10.0


def _render_frames() -> None:
    renderer = _import_renderer()
    runtime_initialized = False

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    camera_eye = CAMERA_EYE
    camera_look_at = CAMERA_LOOK_AT
    camera_up = CAMERA_UP
    scalar_range = SCALAR_RANGE

    if NUM_FRAMES <= 0:
        raise ValueError("num-frames must be a positive integer")

    relative_eye = (
        camera_eye[0] - camera_look_at[0],
        camera_eye[1] - camera_look_at[1],
        camera_eye[2] - camera_look_at[2],
    )
    horizontal_radius = math.hypot(relative_eye[0], relative_eye[2])
    if not math.isfinite(horizontal_radius) or horizontal_radius <= 0.0:
        raise ValueError(
            "camera-eye must have non-zero horizontal distance from the look-at point"
        )

    eye_height = relative_eye[1]
    initial_angle = math.atan2(relative_eye[2], relative_eye[0])

    try:
        renderer.initialize_runtime()
        runtime_initialized = True

        for frame_idx in range(NUM_FRAMES):
            angle = initial_angle + math.tau * frame_idx / NUM_FRAMES
            eye = (
                horizontal_radius * math.cos(angle) + camera_look_at[0],
                eye_height + camera_look_at[1],
                horizontal_radius * math.sin(angle) + camera_look_at[2],
            )
            output_path = OUTPUT_DIR / f"{OUTPUT_PREFIX}_{frame_idx:04d}.png"

            renderer.render(
                plotfile=PLOTFILE,
                width=IMAGE_WIDTH,
                height=IMAGE_HEIGHT,
                variable=VARIABLE,
                output=str(output_path),
                antialiasing=ANTIALIASING,
                log_scale=LOG_SCALE,
                box_transparency=BOX_TRANSPARENCY,
                scalar_range=scalar_range,
                camera_eye=eye,
                camera_look_at=camera_look_at,
                camera_up=camera_up,
                camera_fov_y=FOV_Y,
                camera_near=NEAR_PLANE,
                camera_far=FAR_PLANE,
            )
    finally:
        if runtime_initialized:
            renderer.finalize_runtime()


def main() -> None:
    _render_frames()


if __name__ == "__main__":
    main()
