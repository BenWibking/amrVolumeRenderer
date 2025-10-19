#!/usr/bin/env python3
"""Example render driver that pins the camera and color scale in Python."""

import argparse
import math
import os
from pathlib import Path
import sys
import glob
import amrVolumeRenderer

# CLI parameters:
# --min-level 8 --up-vector 0 0 1 --log-scale --antialiasing 4 --width 1024 --height 1024 --box-transparency 0.985

# Camera parameters (automatic):
#  eye      = (2.545326948, 2.748585939, 6.465749741)
#  look_at  = (0, 0, 0)
#  up       = (0, 0, 1)
#  fov_y    = 45 degrees
#  near     = 0.1000000015
#  far      = 29.89028931

PLOTFILE_GLOB = "./plt*"
VARIABLE = "gasDensity"
MIN_LEVEL = 8
IMAGE_WIDTH = 512
IMAGE_HEIGHT = 512
OUTPUT_DIR = Path("renders")
OUTPUT_PREFIX = "render"
ANTIALIASING = 4
BOX_TRANSPARENCY = 0.985
LOG_SCALE = True
CAMERA_LOOK_AT = (0, 0, 0)
CAMERA_UP = (0, 0, 1)
CAMERA_EYE = (2.545326948, 2.748585939, 6.465749741)
FOV_Y = 45.0
NEAR_PLANE = 0.1
FAR_PLANE = 29.89028931


COLOR_MAP_PHYSICAL = [
    (1.00e-28, 0.02, 0.02, 0.05, 0.00),  # near-empty voids stay dark/transparent
    (2.85e-28, 0.08, 0.10, 0.35, 0.05),  # tenuous gas takes on a muted blue
    (4.54e-28, 0.10, 0.35, 0.45, 0.12),  # filamentary median gains cyan highlight
    (1.05e-27, 0.25, 0.60, 0.40, 0.20),  # bulk mass rendered in teal-green
    (2.22e-27, 0.80, 0.75, 0.25, 0.40),  # overdense knots warm toward yellow
    (2.71e-26, 0.95, 0.55, 0.05, 0.65),  # rare nodes glow amber without clipping
    (4.00e-25, 1.00, 0.95, 0.85, 0.85),  # densest cores roll off into soft white
]

def _render_frames(last_only: bool) -> None:
    runtime_initialized = False
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    camera_eye = CAMERA_EYE
    camera_look_at = CAMERA_LOOK_AT
    camera_up = CAMERA_UP

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
        amrVolumeRenderer.initialize_runtime()
        runtime_initialized = True

        frame_indices = (
            [NUM_FRAMES - 1] if last_only else range(NUM_FRAMES)
        )

        for frame_idx in frame_indices:
            output_path = OUTPUT_DIR / f"{OUTPUT_PREFIX}_{frame_idx:04d}.png"
            if output_path.exists():
                continue
            
            if horizontal_radius > 0.0:
                # Step the camera azimuth as we advance through the plotfiles.
                fraction = frame_idx / NUM_FRAMES
                angle = base_angle + math.pi * fraction
                frame_camera_eye = (
                    camera_look_at[0] + horizontal_radius * math.cos(angle),
                    camera_look_at[1] + rel_eye[1],
                    camera_look_at[2] + horizontal_radius * math.sin(angle),
                )
            else:
                frame_camera_eye = camera_eye

            amrVolumeRenderer.render(
                plotfile=plotfiles[frame_idx],
                min_level=MIN_LEVEL,
                width=IMAGE_WIDTH,
                height=IMAGE_HEIGHT,
                variable=VARIABLE,
                output=str(output_path),
                antialiasing=ANTIALIASING,
                log_scale=LOG_SCALE,
                box_transparency=BOX_TRANSPARENCY,
                color_map=COLOR_MAP_PHYSICAL,
                scalar_range=(COLOR_MAP_PHYSICAL[0][0], COLOR_MAP_PHYSICAL[-1][0]),
                camera_eye=frame_camera_eye,
                camera_look_at=camera_look_at,
                camera_up=camera_up,
                camera_fov_y=FOV_Y,
                camera_near=NEAR_PLANE,
                camera_far=FAR_PLANE,
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
