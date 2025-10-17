#!/usr/bin/env python3
"""Example render driver that pins the camera and color scale in Python."""

import argparse
import math
import os
from pathlib import Path
import sys
import glob
import miniGraphics


# Hard-coded rendering configuration.
PLOTFILE_GLOB = "../quokka-turb-driving/tests/mach_10_v2/plt*"
VARIABLE = "gasDensity"
IMAGE_WIDTH = 512
IMAGE_HEIGHT = 512
OUTPUT_DIR = Path("renders")
OUTPUT_PREFIX = "render"
ANTIALIASING = 4
BOX_TRANSPARENCY = 0.975
LOG_SCALE = True
CAMERA_EYE = (2.0, 1.2, 2.0)
CAMERA_LOOK_AT = (0.5, 0.5, 0.5)
CAMERA_UP = (0.0, 1.0, 0.0)
FOV_Y = 45.0
NEAR_PLANE = 0.1
FAR_PLANE = 10.0

# Physical scalar -> RGBA ramp used for the volume color map. Values are in the
# original field units; they are mapped through math.log when LOG_SCALE is True.
alpha = 0.01
COLOR_MAP_PHYSICAL = [
    (0.0050, 0.00, 0.00, 0.00, 0.00),   # floor: fully transparent
    (0.0052, 0.05, 0.07, 0.20, alpha*0.02),   # 1–5% tail: faint navy sheen
    (0.0061, 0.10, 0.20, 0.40, alpha*0.05),   # 10%: tease out wispy filaments
    (0.074 , 0.05, 0.45, 0.55, alpha*0.12),   # 30%: cool teal accents
    (0.448, 0.20, 0.70, 0.40, alpha*0.18),   # median (~0.45): soft green mid-tones
    (2.71 , 0.95, 0.90, 0.25, alpha*0.26),   # 90%: light golden shocks
    (11.9 , 0.98, 0.55, 0.10, alpha*0.35),   # 99%: semi-transparent orange hotspots
    (25.6 , 0.85, 0.20, 0.05, alpha*0.38),   # near-peak: deep red highlights
    (40.2 , 1.00, 0.95, 0.95, alpha*0.45),   # absolute max: translucent white cap
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


def _render_frames(last_only: bool) -> None:
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

    # Precompute the camera offset so we can orbit around the look-at point.
    rel_eye = (
        camera_eye[0] - camera_look_at[0],
        camera_eye[1] - camera_look_at[1],
        camera_eye[2] - camera_look_at[2],
    )
    horizontal_radius = math.hypot(rel_eye[0], rel_eye[2])
    base_angle = math.atan2(rel_eye[2], rel_eye[0]) if horizontal_radius > 0.0 else 0.0

    try:
        miniGraphics.initialize_runtime()
        runtime_initialized = True

        frame_indices = (
            [NUM_FRAMES - 1] if last_only else range(NUM_FRAMES)
        )

        for frame_idx in frame_indices:
            output_path = OUTPUT_DIR / f"{OUTPUT_PREFIX}_{frame_idx:04d}.ppm"

            if horizontal_radius > 0.0:
                # Step the camera azimuth as we advance through the plotfiles.
                fraction = frame_idx / NUM_FRAMES
                angle = base_angle + 2.0 * math.pi * fraction
                frame_camera_eye = (
                    camera_look_at[0] + horizontal_radius * math.cos(angle),
                    camera_look_at[1] + rel_eye[1],
                    camera_look_at[2] + horizontal_radius * math.sin(angle),
                )
            else:
                frame_camera_eye = camera_eye

            miniGraphics.render(
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
                color_map=color_map,
                scalar_range=scalar_range,
            )
    finally:
        if runtime_initialized:
            miniGraphics.finalize_runtime()


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
