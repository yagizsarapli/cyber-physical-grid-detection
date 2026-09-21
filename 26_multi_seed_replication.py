from pathlib import Path
import subprocess
import sys

import pandas as pd

# ============================================================
# PHASE 2U -- MULTI-SEED REPLICATION OF THE HARD RESULT
# ============================================================
#
# STATUS.md's headline numbers (topology_fusion beats prior_only at
# both detection and localization under realistic attack magnitude)
# come from ONE seed's data draw, with only a within-run bootstrap CI.
# That is not the same as cross-seed replication: a reviewer's first
# question would be "does this hold up with a different random draw
# of the same scenario generator, not just resampling of one draw?"
#
# This script re-runs data generation (22_..._HARD.py) + evaluation
# (23_..._HARD.py) for several independent seeds, sequentially
# (each seed overwrites the same phase2r_hard_graph_*/phase2s_hard_*
# files -- intentional, we only need to KEEP the extracted summary
# numbers per seed, not every raw dataset), and reports mean/std/range
# of the headline metrics across seeds.
# ============================================================

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

SEEDS = [20260812, 42, 777, 2024]  # first is the seed already used throughout
N_REP = 500


def run(cmd):
    print(f"  $ {' '.join(cmd)}")
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-3000:])
        print(r.stderr[-3000:])
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")


def extract_headline(seed):
    det = pd.read_csv(RESULTS / "phase2s_hard_cyber_detection_metrics.csv")
    loc = pd.read_csv(RESULTS / "phase2s_hard_attack_localization_metrics.csv")

    topo_det = det[det["feature_set"] == "topology_fusion"]["balanced_accuracy"].max()
    prior_det = det[det["feature_set"] == "prior_only"]["balanced_accuracy"].max()
    topo_loc = loc[loc["feature_set"] == "topology_fusion"]["top1_accuracy"].max()
    prior_loc = loc[loc["feature_set"] == "prior_only"]["top1_accuracy"].max()

    return {
        "seed": seed,
        "topology_fusion_best_detection_balacc": topo_det,
        "prior_only_best_detection_balacc": prior_det,
        "detection_gap": topo_det - prior_det,
        "topology_fusion_best_localization_top1": topo_loc,
        "prior_only_best_localization_top1": prior_loc,
        "localization_gap": topo_loc - prior_loc,
    }


def main():
    rows = []
    for seed in SEEDS:
        print(f"\n=== SEED {seed} ===")
        run([sys.executable, "22_graph_ready_protected_prior_telemetry_HARD.py",
             "--n-rep", str(N_REP), "--seed", str(seed)])
        run([sys.executable, "23_topology_aware_cyber_physical_localization_HARD.py"])
        row = extract_headline(seed)
        rows.append(row)
        print(f"  detection gap={row['detection_gap']:.4f} | "
              f"localization gap={row['localization_gap']:.4f}")

    df = pd.DataFrame(rows)
    print("\n=== ALL SEEDS ===")
    print(df.to_string(index=False))

    summary = df.drop(columns=["seed"]).agg(["mean", "std", "min", "max"])
    print("\n=== CROSS-SEED SUMMARY ===")
    print(summary.to_string())

    RESULTS.mkdir(exist_ok=True)
    df.to_csv(RESULTS / "phase2u_multiseed_replication.csv", index=False)
    summary.to_csv(RESULTS / "phase2u_multiseed_summary.csv")
    print("\nSaved:\n  results/phase2u_multiseed_replication.csv"
          "\n  results/phase2u_multiseed_summary.csv")


if __name__ == "__main__":
    main()
