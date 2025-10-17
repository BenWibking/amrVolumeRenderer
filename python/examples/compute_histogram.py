#!/usr/bin/env python3
"""Compute a histogram of scalar field values for miniGraphics renders.

This helper loads an AMReX plotfile using the same code path as the renderer
and reports how the scalar values are distributed after any optional log
transforms and normalization. Use it to decide where to place color-map control
points in the normalized [0, 1] domain that the renderer expects.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple


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


def _load_binding() -> Callable[..., Dict[str, object]]:
  try:
    from miniGraphics import compute_histogram  # type: ignore[attr-defined]
    return compute_histogram
  except ModuleNotFoundError:
    _configure_import_path()
    try:
      from miniGraphics import compute_histogram  # type: ignore[attr-defined]
      return compute_histogram
    except ModuleNotFoundError as exc:  # pragma: no cover - defensive
      raise SystemExit(
          "miniGraphics Python extension not found. Build the project or adjust PYTHONPATH."
      ) from exc


def _parse_arguments(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description=("Analyze scalar values from a plotfile to aid color-map tuning.")
  )
  parser.add_argument("plotfile", type=Path, help="Path to the AMReX plotfile")
  parser.add_argument(
      "--variable",
      type=str,
      default=None,
      help="Cell-centered variable to inspect (default: first variable in plotfile)",
  )
  parser.add_argument(
      "--min-level",
      type=int,
      default=0,
      help="Coarsest AMR level to include (default: 0)",
  )
  parser.add_argument(
      "--max-level",
      type=int,
      default=-1,
      help="Finest AMR level to include (-1 renders all levels, default: -1)",
  )
  parser.add_argument(
      "--log-scale",
      action="store_true",
      help="Apply natural-log scaling to positive values before histogramming",
  )
  parser.add_argument(
      "--bins",
      type=int,
      default=128,
      help="Number of histogram bins across the normalized range (default: 128)",
  )
  parser.add_argument(
      "--max-rows",
      type=int,
      default=32,
      help="Maximum number of rows in the printed histogram table (default: 32)",
  )
  parser.add_argument(
      "--no-bars",
      action="store_true",
      help="Disable ASCII bar visualization in the histogram table",
  )
  return parser.parse_args(argv)


def _compute_percentiles(
    counts: Sequence[int], edges: Sequence[float], samples: int, percentiles: Iterable[float]
) -> Dict[float, Optional[float]]:
  if samples <= 0:
    return {p: None for p in percentiles}

  results: Dict[float, Optional[float]] = {}
  sorted_percentiles = sorted(percentiles)
  cumulative = 0
  bin_index = 0

  for percentile in sorted_percentiles:
    target = percentile * (samples - 1)
    while bin_index < len(counts):
      count = counts[bin_index]
      next_cumulative = cumulative + count
      if target < next_cumulative or bin_index == len(counts) - 1:
        if count == 0:
          value = edges[bin_index]
        else:
          position_in_bin = (target - cumulative) / count
          position_in_bin = max(0.0, min(1.0, position_in_bin))
          span = edges[bin_index + 1] - edges[bin_index]
          value = edges[bin_index] + position_in_bin * span
        results[percentile] = value
        break
      cumulative = next_cumulative
      bin_index += 1
    else:
      results[percentile] = edges[-1]

  return results


def _group_bins(
    counts: Sequence[int],
    edges: Sequence[float],
    samples: int,
    max_rows: int,
) -> List[Tuple[float, float, int, float]]:
  if not counts:
    return []
  rows = max(1, max_rows)
  group_size = max(1, math.ceil(len(counts) / rows))
  grouped: List[Tuple[float, float, int, float]] = []
  for start in range(0, len(counts), group_size):
    end = min(start + group_size, len(counts))
    count = sum(counts[start:end])
    range_min = edges[start]
    range_max = edges[end]
    fraction = (count / samples) if samples > 0 else 0.0
    grouped.append((range_min, range_max, count, fraction))
  return grouped


def _format_range(value_range: Sequence[float]) -> str:
  return f"[{value_range[0]:.6g}, {value_range[1]:.6g}]"


def _normalized_to_processed(
    normalized: float,
    normalized_range: Sequence[float],
    processed_range: Optional[Sequence[float]],
) -> Optional[float]:
  if processed_range is None:
    return None
  n_min, n_max = float(normalized_range[0]), float(normalized_range[1])
  n_span = n_max - n_min
  if n_span <= 0.0 or not math.isfinite(n_span):
    return None
  p_min, p_max = float(processed_range[0]), float(processed_range[1])
  return p_min + ((normalized - n_min) / n_span) * (p_max - p_min)


def _normalized_to_physical(
    normalized: float,
    normalized_range: Sequence[float],
    processed_range: Optional[Sequence[float]],
    original_range: Optional[Sequence[float]],
    log_scale: bool,
) -> Optional[float]:
  processed_value = _normalized_to_processed(
      normalized, normalized_range, processed_range
  )
  if processed_value is None:
    if not log_scale and isinstance(original_range, Sequence):
      # Fall back to original range when no processing occurred.
      o_min, o_max = float(original_range[0]), float(original_range[1])
      span = o_max - o_min
      if span <= 0.0 or not math.isfinite(span):
        return None
      n_min, n_max = float(normalized_range[0]), float(normalized_range[1])
      n_span = n_max - n_min
      if n_span <= 0.0 or not math.isfinite(n_span):
        return None
      return o_min + ((normalized - n_min) / n_span) * span
    return None
  if not log_scale:
    return processed_value
  return math.exp(processed_value)


def _print_summary(
    args: argparse.Namespace,
    histogram: Dict[str, object],
    counts: Sequence[int],
    edges_normalized: Sequence[float],
) -> None:
  samples = int(histogram.get("samples", 0))
  normalized_range = histogram.get("normalized_range")
  processed_range = histogram.get("processed_range")
  original_range = histogram.get("original_range")

  print(f"Plotfile: {args.plotfile}")
  if args.variable is not None:
    print(f"Variable: {args.variable}")
  levels = f"{args.min_level}..{args.max_level if args.max_level >= 0 else 'max'}"
  print(f"Levels: {levels}")
  print(f"Log scale: {'yes' if args.log_scale else 'no'}")
  print(f"Samples: {samples}")
  if isinstance(normalized_range, Sequence):
    print(f"Normalized range: {_format_range(normalized_range)}")
  if isinstance(processed_range, Sequence):
    print(f"Processed range: {_format_range(processed_range)}")
  if isinstance(original_range, Sequence):
    label = "Original range (pre-log)" if args.log_scale else "Original range"
    print(f"{label}: {_format_range(original_range)}")

  percentiles_to_report = [0.01, 0.05, 0.1, 0.5, 0.9, 0.95, 0.99]
  percentile_values = _compute_percentiles(
      counts, edges_normalized, samples, percentiles_to_report
  )

  def _format_percentile_line(p: float) -> str:
    normalized_value = percentile_values[p]
    if normalized_value is None:
      return f"{p*100:5.1f}%  --"
    processed_value = _normalized_to_processed(
        normalized_value, normalized_range, processed_range  # type: ignore[arg-type]
    )
    physical_value = _normalized_to_physical(
        normalized_value,
        normalized_range,  # type: ignore[arg-type]
        processed_range,
        original_range,
        args.log_scale,
    )
    pieces = [f"{p*100:5.1f}%"]
    if physical_value is not None:
      pieces.append(f"{physical_value:12.6g}")
    else:
      pieces.append(" " * 12)
    if processed_value is not None and args.log_scale:
      pieces.append(f"{processed_value:10.6f}")  # log-space diagnostic
    pieces.append(f"(normalized {normalized_value:6.3f})")
    return "  ".join(pieces)

  header = "Percentiles (%, physical density"
  if args.log_scale:
    header += ", log-density"
  header += ", normalized)"
  print()
  print(header)
  for percentile in percentiles_to_report:
    print(_format_percentile_line(percentile))

  grouped = _group_bins(counts, edges_normalized, samples, args.max_rows)
  if not grouped:
    return

  print()
  print("Histogram (physical density range, count, fraction)")
  if args.no_bars:
    bar_scale = 0.0
  else:
    bar_scale = max((g[2] for g in grouped), default=0)
  bar_width = 40
  for lower, upper, count, fraction in grouped:
    lower_phys = _normalized_to_physical(
        lower,
        normalized_range,  # type: ignore[arg-type]
        processed_range,
        original_range,
        args.log_scale,
    )
    upper_phys = _normalized_to_physical(
        upper,
        normalized_range,  # type: ignore[arg-type]
        processed_range,
        original_range,
        args.log_scale,
    )
    if lower_phys is None or upper_phys is None:
      range_text = f"{lower:6.3f} - {upper:6.3f}"
    else:
      range_text = f"{lower_phys:9.3e} - {upper_phys:9.3e}"
    fraction_text = f"{fraction * 100:6.2f}%"
    if bar_scale > 0 and count > 0:
      filled = max(1, int(round((count / bar_scale) * bar_width)))
      bar = "#" * filled
    else:
      bar = ""
    print(f"{range_text}  {count:10d}  {fraction_text}  {bar}")


def main(argv: Optional[Sequence[str]] = None) -> int:
  args = _parse_arguments(argv)

  if args.bins <= 0:
    print("--bins must be a positive integer.", file=sys.stderr)
    return 1
  if args.max_rows <= 0:
    print("--max-rows must be a positive integer.", file=sys.stderr)
    return 1

  compute_histogram = _load_binding()

  histogram = compute_histogram(
      plotfile=str(args.plotfile),
      variable=args.variable,
      min_level=args.min_level,
      max_level=args.max_level,
      log_scale=args.log_scale,
      bins=args.bins,
  )

  counts = list(histogram["counts"])
  normalized_range = histogram["normalized_range"]
  if not isinstance(normalized_range, Sequence) or len(normalized_range) != 2:
    print("Histogram did not provide a normalized range.", file=sys.stderr)
    return 1
  norm_min, norm_max = float(normalized_range[0]), float(normalized_range[1])
  norm_span = norm_max - norm_min
  if norm_span <= 0.0 or not math.isfinite(norm_span):
    print("Histogram reported an invalid normalized range.", file=sys.stderr)
    return 1

  edges_normalized = [
      norm_min + norm_span * i / len(counts) for i in range(len(counts) + 1)
  ]
  _print_summary(args, histogram, counts, edges_normalized)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
