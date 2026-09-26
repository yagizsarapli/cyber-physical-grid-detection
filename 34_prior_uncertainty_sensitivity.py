from pathlib import Path
import argparse
import json
import subprocess
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
DEFAULT_SEEDS = [
    20260812, 42, 777, 2024,
    3, 11, 19, 37,
    53, 71, 97, 131,
    163, 197, 229, 251,
]


def run(cmd):
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        print(proc.stdout[-4000:])
        print(proc.stderr[-4000:], file=sys.stderr)
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")


def extract(seed, prior_sigma, prior_clip):
    det = pd.read_csv(RESULTS / "phase2s_hard_cyber_detection_metrics.csv")
    loc = pd.read_csv(RESULTS / "phase2s_hard_attack_localization_metrics.csv")

    topo_det = float(
        det.loc[det["feature_set"] == "topology_fusion", "balanced_accuracy"].max()
    )
    rpp_det = float(
        det.loc[det["feature_set"] == "residual_plus_prior", "balanced_accuracy"].max()
    )
    topo_loc = float(
        loc.loc[loc["feature_set"] == "topology_fusion", "top1_accuracy"].max()
    )
    rpp_loc = float(
        loc.loc[loc["feature_set"] == "residual_plus_prior", "top1_accuracy"].max()
    )

    return {
        "seed": int(seed),
        "prior_sigma": float(prior_sigma),
        "prior_clip": float(prior_clip),
        "topology_fusion_detection": topo_det,
        "residual_plus_prior_detection": rpp_det,
        "detection_gap": topo_det - rpp_det,
        "topology_fusion_localization": topo_loc,
        "residual_plus_prior_localization": rpp_loc,
        "localization_gap": topo_loc - rpp_loc,
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Sensitivity test for the 5-bus topology ablation under a "
            "different prior-forecast uncertainty."
        )
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=DEFAULT_SEEDS,
    )
    parser.add_argument("--n-rep", type=int, default=500)
    parser.add_argument("--prior-sigma", type=float, default=0.05)
    parser.add_argument(
        "--prior-clip",
        type=float,
        default=0.0,
        help="<=0 disables clipping; default matches the unclipped IEEE prior convention.",
    )
    args = parser.parse_args()

    rows = []
    for seed in args.seeds:
        run([
            sys.executable,
            "22_graph_ready_protected_prior_telemetry_HARD.py",
            "--n-rep",
            str(args.n_rep),
            "--seed",
            str(seed),
            "--prior-sigma",
            str(args.prior_sigma),
            "--prior-clip",
            str(args.prior_clip),
        ])
        run([
            sys.executable,
            "23_topology_aware_cyber_physical_localization_HARD.py",
        ])
        row = extract(seed, args.prior_sigma, args.prior_clip)
        rows.append(row)
        print("SENSITIVITY_RESULT_JSON=" + json.dumps(row, sort_keys=True), flush=True)

    out = pd.DataFrame(rows)
    RESULTS.mkdir(exist_ok=True)
    tag = f"prior_sigma_{args.prior_sigma:.3f}".replace(".", "p")
    out_path = RESULTS / f"phase2aa_5bus_{tag}_sensitivity.csv"
    out.to_csv(out_path, index=False)

    print("\n=== PRIOR-UNCERTAINTY SENSITIVITY ===")
    print(out.to_string(index=False))
    print("\nGap summary:")
    print(out[["detection_gap", "localization_gap"]].agg(["mean", "std", "min", "max"]).to_string())
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
