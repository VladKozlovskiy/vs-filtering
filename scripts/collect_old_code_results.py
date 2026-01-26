#!/usr/bin/env python3
"""Collect old code verification results and compare with expected values."""

from pathlib import Path

import pandas as pd
import yaml

# Expected results from old code (Fedor)
OLD_CODE_EXPECTED = {
    "old_code_1_squeeze_avg_rgb": {"pearson": -0.637, "spearman": 0.646, "auc": 0.796},
    "old_code_2_alex_avg_rgb": {"pearson": -0.610, "spearman": 0.623, "auc": 0.796},
    "old_code_3_squeeze_avg_gray": {"pearson": -0.603, "spearman": 0.610, "auc": 0.784},
    "old_code_4_vgg_avg_rgb": {"pearson": -0.586, "spearman": 0.597, "auc": 0.788},
    "old_code_5_vgg_lin_rgb": {"pearson": -0.571, "spearman": 0.590, "auc": 0.791},
}


def find_latest_result(outputs_dir: Path, name_prefix: str) -> dict | None:
    """Find the latest result file for a given experiment name prefix."""
    candidates = []
    for d in outputs_dir.iterdir():
        if d.is_dir() and d.name.startswith(name_prefix):
            results_file = d / "results.yaml"
            if results_file.exists():
                candidates.append((d.stat().st_mtime, results_file))

    if not candidates:
        return None

    # Get the most recent one
    candidates.sort(reverse=True)
    with open(candidates[0][1]) as f:
        return yaml.safe_load(f)


def main():
    outputs_dir = Path("outputs")

    rows = []
    for exp_name, expected in OLD_CODE_EXPECTED.items():
        row = {"experiment": exp_name}

        # Expected values
        row["old_pearson"] = expected["pearson"]
        row["old_spearman"] = expected["spearman"]
        row["old_auc"] = expected["auc"]

        # Current dist results
        dist_result = find_latest_result(outputs_dir, exp_name)
        if dist_result:
            row["new_pearson"] = dist_result.get("LPIPS_Pearson", None)
            row["new_spearman"] = dist_result.get("LPIPS_Spearman", None)
        else:
            row["new_pearson"] = None
            row["new_spearman"] = None

        # Current class results
        class_result = find_latest_result(outputs_dir, f"{exp_name}_class")
        if class_result:
            row["new_auc"] = class_result.get("LPIPS_MulticlassAUC", None)
        else:
            row["new_auc"] = None

        rows.append(row)

    df = pd.DataFrame(rows)

    # Reorder columns
    df = df[
        [
            "experiment",
            "old_pearson",
            "new_pearson",
            "old_spearman",
            "new_spearman",
            "old_auc",
            "new_auc",
        ]
    ]

    # Round for display
    for col in df.columns:
        if col != "experiment" and df[col].dtype == float:
            df[col] = df[col].round(4)

    print("=" * 100)
    print("COMPARISON: Old Code vs New Code")
    print("=" * 100)
    print(df.to_string(index=False))
    print()

    # Save to CSV
    output_path = Path("old_code_comparison.csv")
    df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")

    # Check differences
    print("\n" + "=" * 100)
    print("DIFFERENCES (new - old)")
    print("=" * 100)

    for _, row in df.iterrows():
        exp = row["experiment"]
        diffs = []

        if row["new_pearson"] is not None:
            diff = abs(row["new_pearson"] - row["old_pearson"])
            diffs.append(f"pearson: {diff:.4f}")

        if row["new_spearman"] is not None:
            diff = abs(row["new_spearman"] - row["old_spearman"])
            diffs.append(f"spearman: {diff:.4f}")

        if row["new_auc"] is not None:
            diff = abs(row["new_auc"] - row["old_auc"])
            diffs.append(f"auc: {diff:.4f}")

        if diffs:
            print(f"{exp}: {', '.join(diffs)}")
        else:
            print(f"{exp}: NO RESULTS YET")


if __name__ == "__main__":
    main()
