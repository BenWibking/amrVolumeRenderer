#!/usr/bin/env python3
"""Example render driver that pins the camera and color scale in Python."""

import argparse
import math
import os
from pathlib import Path
import sys
import glob
import amrVolumeRenderer


# Hard-coded rendering configuration.
PLOTFILE_GLOB = "../quokka/tests/plt*"
VARIABLE = "gasDensity"
IMAGE_WIDTH = 512
IMAGE_HEIGHT = 512
OUTPUT_DIR = Path("renders")
OUTPUT_PREFIX = "render"
ANTIALIASING = 4
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
# Increase contrast further by suppressing diffuse fill and reserving opacity
# for dense structures and shocks.
COLOR_MAP_PHYSICAL = [
    (0.0050, 0.00, 0.00, 0.00, 0.00),   # floor: fully transparent
    (0.0058, 0.01, 0.03, 0.15, 0.00),   # negligible halo
    (0.0075, 0.05, 0.10, 0.28, 0.006),  # hint of structure
    (0.0120, 0.12, 0.30, 0.58, 0.014),  # thin teal filaments start appearing
    (0.0300, 0.18, 0.52, 0.74, 0.028),  # emphasize coherent sheets
    (0.0800, 0.14, 0.72, 0.82, 0.050),  # cooler cyan mid-tones, still restrained
    (0.2200, 0.26, 0.88, 0.62, 0.075),  # bright greens on dense ridges
    (0.6500, 0.90, 0.68, 0.24, 0.100),  # warm transition into shocks
    (1.8000, 1.00, 0.46, 0.10, 0.150),  # glowing oranges for strong shocks
    (6.5000, 1.00, 0.24, 0.03, 0.800),  # tight crimson highlights
    (22.0000, 0.98, 0.86, 0.78, 0.90),  # soften approach to the clip region
    (1.000e3, 1.00, 0.96, 0.94, 1.00),  # controlled roll-off into white
]


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _length(vec):
    return math.sqrt(_dot(vec, vec))


def _normalize(vec):
    length = _length(vec)
    if length <= 0.0:
        return None
    return (vec[0] / length, vec[1] / length, vec[2] / length)


def _rotate_about_axis(vec, axis, angle):
    """Rotate vec around the provided unit axis by angle radians."""
    cos_angle = math.cos(angle)
    sin_angle = math.sin(angle)
    dot = _dot(vec, axis)
    cross = (
        axis[1] * vec[2] - axis[2] * vec[1],
        axis[2] * vec[0] - axis[0] * vec[2],
        axis[0] * vec[1] - axis[1] * vec[0],
    )
    return (
        vec[0] * cos_angle + cross[0] * sin_angle + axis[0] * dot * (1.0 - cos_angle),
        vec[1] * cos_angle + cross[1] * sin_angle + axis[1] * dot * (1.0 - cos_angle),
        vec[2] * cos_angle + cross[2] * sin_angle + axis[2] * dot * (1.0 - cos_angle),
    )


def _render_frames(last_only: bool) -> None:
    runtime_initialized = False
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    camera_eye = CAMERA_EYE
    camera_look_at = CAMERA_LOOK_AT
    camera_up = CAMERA_UP

    scalar_range = (
        (COLOR_MAP_PHYSICAL[0][0], COLOR_MAP_PHYSICAL[-1][0])
        if COLOR_MAP_PHYSICAL
        else None
    )

    plotfiles = sorted(glob.glob(PLOTFILE_GLOB))
    NUM_FRAMES = len(plotfiles)
    if NUM_FRAMES <= 0:
        raise ValueError("num-frames must be a positive integer")

    # Precompute the camera offset so we can orbit around the look-at point.
    rel_eye = (
        camera_eye[0] - camera_look_at[0],
        camera_eye[1] - camera_look_at[1],
        camera_eye[2] - camera_look_at[2],
    )
    unit_up = _normalize(camera_up)
    if unit_up is None:
        raise ValueError("camera_up must be a non-zero vector.")
    rel_eye_parallel = _dot(rel_eye, unit_up)
    rel_eye_perp = (
        rel_eye[0] - unit_up[0] * rel_eye_parallel,
        rel_eye[1] - unit_up[1] * rel_eye_parallel,
        rel_eye[2] - unit_up[2] * rel_eye_parallel,
    )
    orbit_radius = _length(rel_eye_perp)

    try:
        amrVolumeRenderer.initialize_runtime()
        runtime_initialized = True

        frame_indices = (
            [NUM_FRAMES - 1] if last_only else range(NUM_FRAMES)
        )

        for frame_idx in frame_indices:
            output_path = OUTPUT_DIR / f"{OUTPUT_PREFIX}_{frame_idx:04d}.png"

            if orbit_radius > 0.0:
                # Step the camera azimuth as we advance through the plotfiles.
                fraction = frame_idx / NUM_FRAMES
                angle = math.pi * fraction
                rotated_rel_eye = _rotate_about_axis(rel_eye, unit_up, angle)
                frame_camera_eye = (
                    camera_look_at[0] + rotated_rel_eye[0],
                    camera_look_at[1] + rotated_rel_eye[1],
                    camera_look_at[2] + rotated_rel_eye[2],
                )
            else:
                frame_camera_eye = camera_eye

            amrVolumeRenderer.render(
                plotfile=plotfiles[frame_idx],
                width=IMAGE_WIDTH,
                height=IMAGE_HEIGHT,
                variable=VARIABLE,
                output=str(output_path),
                antialiasing=ANTIALIASING,
                log_scale=LOG_SCALE,
                box_transparency=BOX_TRANSPARENCY,
                camera_eye=frame_camera_eye,
                camera_look_at=camera_look_at,
                camera_up=camera_up,
                camera_fov_y=FOV_Y,
                camera_near=NEAR_PLANE,
                camera_far=FAR_PLANE,
                color_map=COLOR_MAP_PHYSICAL,
                scalar_range=scalar_range,
            )
    finally:
        if runtime_initialized:
            amrVolumeRenderer.finalize_runtime()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render a series of plotfiles with a fixed camera."
    )
    parser.add_argument(
        "--last-only",
        action="store_true",
        help="Render only the final plotfile in the sequence.",
    )
    args = parser.parse_args()

    _render_frames(last_only=args.last_only)


if __name__ == "__main__":
    main()
