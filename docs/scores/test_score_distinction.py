#!/usr/bin/env python
"""
Test score module on sample patches and visualize distinctions.

Loads sample patches, computes scores for each, and generates a report
showing how well the score module distinguishes between different patch types.
"""

"""
python docs/scores/test_score_distinction.py --patches docs/scores/sample_patches
"""

import sys
from pathlib import Path

# Ensure the repository root is on sys.path so `src` is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import yaml
import numpy as np
from PIL import Image

from src.utils.embedder import Embedder
from src.utils.patch_scores import *


def load_config(config_path):
    """Load score configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def select_score_module_from_config(config, embedder):
    """Build score module from configuration dict."""
    # Check if using predefined engine

    if "module_name" not in config:
        return PATCH_SCORE_MODULES["text_align_score"](embedder=embedder)

    for key in config.get("params", {}):
        if (
            key
            not in PATCH_SCORE_MODULES[
                config["module_name"]
            ].__init__.__code__.co_varnames
        ):
            raise ValueError(
                f"Unknown parameter '{key}' for score module '{config['module_name']}'"
            )

    return PATCH_SCORE_MODULES[config["module_name"]](
        embedder=embedder, **config.get("params", {})
    )


def compute_patch_scores(patches_dir, score_module, embedder):
    """
    Compute scores for all sample patches.

    Returns dict mapping patch_name -> (scores)
    """
    patches_dir = Path(patches_dir)

    # Load patch info if available
    info_path = patches_dir / "patch_info.json"
    patch_info = {}
    if info_path.exists():
        import json

        with open(info_path, "r") as f:
            patch_info = json.load(f)

    results = {}
    print("\nComputing scores for sample patches:")
    print("-" * 60)

    for patch_file in sorted(patches_dir.glob("*.png")):
        patch_name = patch_file.stem
        patch = Image.open(patch_file)

        # Create artificial child patches with variations
        child_patches = [
            patch.crop((0, 0, patch.width // 2, patch.height // 2)),  # Top-left quarter
            patch.crop(
                (patch.width // 2, 0, patch.width, patch.height // 2)
            ),  # Top-right quarter
            patch.crop(
                (0, patch.height // 2, patch.width // 2, patch.height)
            ),  # Bottom-left quarter
            patch.crop(
                (patch.width // 2, patch.height // 2, patch.width, patch.height)
            ),  # Bottom-right quarter
        ]

        # Compute both STOP (action=0) and ZOOM (action=1) scores
        s_stop = score_module.compute_stop(
            parent_patch=patch,
        )

        s_zoom = score_module.compute_zoom(
            parent_patch=patch,
            child_patches=child_patches,
        )

        results[patch_name] = {
            "stop": float(s_stop),
            "zoom": float(s_zoom),
            "delta": float(s_zoom) - float(s_stop),
            "info": patch_info.get(patch_name, {}),
        }
        print(
            f"  {patch_name:20s} → stop = {s_stop:+.4f}, zoom = {s_zoom:+.4f}, delta = {s_zoom - s_stop:+.4f}"
        )

    print("-" * 60)
    return results


def generate_report(scores, score_module, output_path):
    """Generate text report with statistics."""
    # Guard: empty scores -> write a short message and exit
    if not scores:
        with open(output_path, "w") as f:
            f.write("SCORE DISTINCTION ANALYSIS REPORT\n")
            f.write("No patches were processed; no report generated.\n")
        print(f"No patches found — wrote empty report to: {output_path}")
        return
    stop_scores = {name: data["stop"] for name, data in scores.items()}
    zoom_scores = {name: data["zoom"] for name, data in scores.items()}
    delta_scores = {name: data["delta"] for name, data in scores.items()}

    sorted_by_zoom = sorted(zoom_scores.items(), key=lambda x: x[1], reverse=True)

    min_stop = min(stop_scores.values())
    max_stop = max(stop_scores.values())
    mean_stop = np.mean(list(stop_scores.values()))
    std_stop = np.std(list(stop_scores.values()))

    min_zoom = min(zoom_scores.values())
    max_zoom = max(zoom_scores.values())
    mean_zoom = np.mean(list(zoom_scores.values()))
    std_zoom = np.std(list(zoom_scores.values()))

    min_delta = min(delta_scores.values())
    max_delta = max(delta_scores.values())
    mean_delta = np.mean(list(delta_scores.values()))
    std_delta = np.std(list(delta_scores.values()))

    with open(output_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("SCORE DISTINCTION ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n\n")

        # Score engine configuration
        f.write("Score Engine Configuration:\n")
        f.write("-" * 80 + "\n")
        f.write("\n")

        f.write("Summary Statistics:\n")
        f.write("-" * 80 + "\n")
        f.write(f"  Number of samples: {len(scores)}\n")
        f.write("  STOP scores:\n")
        f.write(f"    Range: {min_stop:.4f} to {max_stop:.4f}\n")
        f.write(f"    Mean: {mean_stop:.4f}  Std: {std_stop:.4f}\n")
        f.write("  ZOOM scores:\n")
        f.write(f"    Range: {min_zoom:.4f} to {max_zoom:.4f}\n")
        f.write(f"    Mean: {mean_zoom:.4f}  Std: {std_zoom:.4f}\n")
        f.write("  ZOOM - STOP (delta):\n")
        f.write(f"    Range: {min_delta:.4f} to {max_delta:.4f}\n")
        f.write(f"    Mean: {mean_delta:.4f}  Std: {std_delta:.4f}\n\n")

        f.write("Scores (sorted by ZOOM desc):\n")
        f.write("-" * 80 + "\n")
        for name, zoom_value in sorted_by_zoom:
            stop_value = stop_scores[name]
            delta = delta_scores[name]
            deviation = (zoom_value - mean_zoom) / std_zoom if std_zoom > 0 else 0
            f.write(
                f"  {name:20s} ZOOM={zoom_value:+.4f}  STOP={stop_value:+.4f}  DELTA={delta:+.4f}  ({deviation:+.2f}σ)\n"
            )

            # Show patch metadata if available
            info = scores[name].get("info", {})
            if info:
                f.write(f"      Level: {info.get('level', 'N/A')}, ")
                f.write(f"Pos: ({info.get('x', 'N/A')}, {info.get('y', 'N/A')}), ")
                f.write(f"Density: {info.get('density', 0):.3f}, ")
                f.write(f"Variance: {info.get('variance', 0):.1f}\n")

        f.write("\n")

        # Category-level summary tables (STOP and ZOOM)
        f.write("Category Summary (Mean Scores by Action):\n")
        f.write("-" * 80 + "\n")

        # Group by category for STOP and ZOOM
        categories_stop = {}
        categories_zoom = {}
        for name, data in scores.items():
            category = name.rsplit("_", 1)[0]
            categories_stop.setdefault(category, []).append((name, data["stop"]))
            categories_zoom.setdefault(category, []).append((name, data["zoom"]))

        def compute_cat_stats(group):
            stats = {}
            for cat, name_value_pairs in group.items():
                values = [v for _, v in name_value_pairs]
                stats[cat] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(min(values)),
                    "max": float(max(values)),
                    "count": len(values),
                    "values": sorted(
                        name_value_pairs, key=lambda x: int(x[0].rsplit("_", 1)[1])
                    ),
                }
            return stats

        cat_stats_stop = compute_cat_stats(categories_stop)
        cat_stats_zoom = compute_cat_stats(categories_zoom)

        # Print STOP table
        f.write("  STOP scores by category:\n")
        f.write(
            f"  {'Category':<12} {'Count':<7} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10} Individual Values\n"
        )
        f.write(f"  {'-'*12} {'-'*7} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*50}\n")
        category_order = ["dense", "sparse", "stained", "blank"]
        for cat in category_order:
            if cat not in cat_stats_stop:
                continue
            s = cat_stats_stop[cat]
            individual = "  ".join([f"{v:+.4f}" for _, v in s["values"]])
            f.write(
                f"  {cat:<12} {s['count']:<7} {s['mean']:>+9.4f}  {s['std']:>9.4f}  {s['min']:>+9.4f}  {s['max']:>+9.4f}  {individual}\n"
            )

        f.write("\n")

        # Print ZOOM table
        f.write("  ZOOM scores by category:\n")
        f.write(
            f"  {'Category':<12} {'Count':<7} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10} Individual Values\n"
        )
        f.write(f"  {'-'*12} {'-'*7} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*50}\n")
        for cat in category_order:
            if cat not in cat_stats_zoom:
                continue
            s = cat_stats_zoom[cat]
            individual = "  ".join([f"{v:+.4f}" for _, v in s["values"]])
            f.write(
                f"  {cat:<12} {s['count']:<7} {s['mean']:>+9.4f}  {s['std']:>9.4f}  {s['min']:>+9.4f}  {s['max']:>+9.4f}  {individual}\n"
            )

        f.write("\n")

        # Delta table (ZOOM - STOP) by category
        f.write("Delta (ZOOM - STOP) by category:\n")
        f.write(
            f"  {'Category':<12} {'CountΔ':<7} {'MeanΔ':<10} {'StdΔ':<10} {'MinΔ':<10} {'MaxΔ':<10} Individual Deltas\n"
        )
        f.write(f"  {'-'*12} {'-'*7} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*50}\n")

        # consider union of categories present in either table
        all_cats = []
        for cat in category_order:
            if cat in cat_stats_stop or cat in cat_stats_zoom:
                all_cats.append(cat)

        for cat in all_cats:
            s_stop = cat_stats_stop.get(cat, None)
            s_zoom = cat_stats_zoom.get(cat, None)

            # Prepare numeric deltas (use 0.0 for missing numeric summaries)
            count_stop = s_stop["count"] if s_stop is not None else 0
            count_zoom = s_zoom["count"] if s_zoom is not None else 0
            delta_count = count_zoom - count_stop

            mean_stop = s_stop["mean"] if s_stop is not None else 0.0
            mean_zoom = s_zoom["mean"] if s_zoom is not None else 0.0
            delta_mean = mean_zoom - mean_stop

            std_stop = s_stop["std"] if s_stop is not None else 0.0
            std_zoom = s_zoom["std"] if s_zoom is not None else 0.0
            delta_std = std_zoom - std_stop

            min_stop = s_stop["min"] if s_stop is not None else 0.0
            min_zoom = s_zoom["min"] if s_zoom is not None else 0.0
            delta_min = min_zoom - min_stop

            max_stop = s_stop["max"] if s_stop is not None else 0.0
            max_zoom = s_zoom["max"] if s_zoom is not None else 0.0
            delta_max = max_zoom - max_stop

            # Compute per-sample deltas where both stop and zoom values exist
            vals_stop = {
                n: v for n, v in (s_stop["values"] if s_stop is not None else [])
            }
            vals_zoom = {
                n: v for n, v in (s_zoom["values"] if s_zoom is not None else [])
            }
            all_names = sorted(
                set(list(vals_stop.keys()) + list(vals_zoom.keys())),
                key=lambda x: (
                    int(x.rsplit("_", 1)[1])
                    if ("_" in x and x.rsplit("_", 1)[1].isdigit())
                    else x
                ),
            )

            indiv = []
            for name in all_names:
                if name in vals_stop and name in vals_zoom:
                    d = vals_zoom[name] - vals_stop[name]
                    indiv.append(f"{d:+.4f}")
                else:
                    indiv.append(" NA ")

            individual = "  ".join(indiv)

            f.write(
                f"  {cat:<12} {delta_count:<7d} {delta_mean:>+9.4f}  {delta_std:>9.4f}  {delta_min:>+9.4f}  {delta_max:>+9.4f}  {individual}\n"
            )

        f.write("\n")

        # Distinction quality assessment
        f.write("Distinction Quality Assessment:\n")
        f.write("-" * 80 + "\n")

        spread = max_zoom - min_zoom
        f.write(f"  Overall spread (ZOOM): {spread:.4f}\n\n")

        if spread < 0.1:
            f.write("  ⚠️  POOR: Very low spread (<0.1)\n")
            f.write("     Score function barely distinguishes between patches\n\n")
            f.write("  💡 Suggestions to increase spread:\n")
            f.write("     - Increase InfoGainScore weight (try 5.0 or 10.0)\n")
            f.write("     - Decrease PatchCost (try 0.1 or 0.2)\n")
            f.write("     - Add TissuePresenceScore with high weight\n")
        elif spread < 0.5:
            f.write("  ⚠️  FAIR: Low spread (0.1-0.5)\n")
            f.write("     Weak distinction between patch types\n\n")
            f.write("  💡 Suggestions to increase spread:\n")
            f.write("     - Increase InfoGainScore weight (try 5.0 or 8.0)\n")
            f.write("     - Decrease PatchCost (try 0.2 or 0.3)\n")
            f.write(
                "     - Current spread mainly from InfoGain variance in child patches\n"
            )
        elif spread < 1.0:
            f.write("  ✓ GOOD: Moderate spread (0.5-1.0)\n")
            f.write("     Reasonable distinction between patch types\n")
        else:
            f.write("  ✓✓ EXCELLENT: High spread (>1.0)\n")
            f.write("     Strong distinction between patch types\n")

        f.write("\n")

        # Check for expected patterns
        f.write("Expected Pattern Check:\n")
        f.write("-" * 80 + "\n")

        if "blank" in zoom_scores:
            if zoom_scores["blank"] < mean_zoom:
                f.write("  ✓ Blank patch gets below-average score (good)\n")
            else:
                f.write("  ⚠️  Blank patch gets above-average score (unexpected)\n")
        if "dense_tissue" in zoom_scores:
            if zoom_scores["dense_tissue"] > mean_zoom:
                f.write("  ✓ Dense tissue gets above-average score (good)\n")
            else:
                f.write("  ⚠️  Dense tissue gets above-average score (unexpected)\n")
        if "complex" in zoom_scores:
            if zoom_scores["complex"] > mean_zoom:
                f.write("  ✓ Complex patch gets above-average score (good)\n")
            else:
                f.write("  ⚠️  Complex patch gets above-average score (unexpected)\n")
        f.write("\n")
        f.write("=" * 80 + "\n")

    print(f"Report saved to: {output_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Test score function on sample patches"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="./docs/scores/score_config.yaml",
        help="Path to score configuration YAML",
    )
    parser.add_argument(
        "--patches",
        type=str,
        default="./docs/scores/sample_patches",
        help="Directory containing sample patches",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./docs/scores/results",
        help="Output directory for results",
    )

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load PLIP for embeddings
    print("Loading PLIP model...")

    embedder = Embedder(img_backend="plip")

    # Load configuration
    print(f"Loading configuration from: {args.config}")
    config = load_config(args.config)

    # Build score module
    print("Building score module...")
    score_module = select_score_module_from_config(config, embedder)

    # Compute scores
    scores = compute_patch_scores(args.patches, score_module, embedder)

    # Generate report
    report_path = output_dir / "score_report.txt"
    generate_report(scores, score_module, report_path)
    print(f"\n✓ Analysis complete!")
    print(f"  Results saved to: {output_dir}/")


if __name__ == "__main__":
    main()
