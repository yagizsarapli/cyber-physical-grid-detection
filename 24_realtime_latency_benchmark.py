from pathlib import Path
import time
import importlib.util
import warnings

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")

# ============================================================
# PHASE 2T
# END-TO-END DECISION LATENCY VS. PROTECTION-GRADE BUDGET
# ============================================================
#
# Purpose
# -------
# Phases 2A-2S (and the 2Q/2R "hard" stress variant) validated WHAT
# the cyber-vs-physical classifier can tell us and answered it
# honestly. This phase asks a different question: how FAST can it
# tell us, end to end, per decision?
#
# A protective relay must clear a fault within about one power-cycle.
# This testbed is an unspecified-frequency pandapower network; 50 Hz
# is pandapower's own default and matches the 0.4 kV / European
# convention used elsewhere in this project, so we use it as the
# budget reference (1 cycle = 20 ms). This is stated as an assumption,
# not measured from the model.
#
# Per-cycle pipeline (network + measurement-model construction is a
# ONE-TIME setup cost in any real deployment and is excluded from the
# timed loop, matching how a relay would actually operate):
#
#   1) WLS state estimation on the newly arrived measurement vector.
#   2) Feature computation (per-node residual/innovation score +
#      simple neighbor aggregation over the fixed 5-node/4-edge
#      graph). This is a simplified but representative stand-in for
#      the full Phase 2S/2R feature pipeline -- the underlying
#      operations (a handful of subtractions, means, maxes over <=5
#      numbers) are what matters for cost, not the exact formula.
#   3) Trained classifier .predict() on a single row.
#
# Single-variable question: does step 1, 2, or 3 blow the budget --
# and if the total does, which stage should be optimized first?
# ============================================================


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"

CYCLE_MS = 1000.0 / 50.0  # 20 ms, 50 Hz assumption (see header)
N_TRIALS = 60
RANDOM_SEED = 20260920


def load_module(filename, module_name):
    path = ROOT / filename
    if not path.exists():
        raise FileNotFoundError(f"Required project file not found: {path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


phase21 = load_module(
    "21_nonlinear_stealth_fdia_bdd_benchmark_HARD.py",
    "phase21_hard_for_2t",
)
topology = phase21.topology

SLOW_PATH = DATA / "phase2e_7day_operating_dataset.csv"


def build_adjacency(net):
    adjacency = {int(b): set() for b in net.bus.index}
    for _, ln in net.line.iterrows():
        f = int(ln["from_bus"])
        t = int(ln["to_bus"])
        adjacency[f].add(t)
        adjacency[t].add(f)
    return adjacency


def timed_feature_step(node_ids, adjacency, norm_res, innov):
    """Representative (not byte-identical to Phase 2S/2R) per-node
    residual/innovation summary + neighbor mean/max aggregation over
    the fixed small graph. See header note."""
    out = {}
    for b in node_ids:
        nbs = sorted(adjacency[b])
        nb_innov = innov[nbs]
        nb_res = norm_res[nbs]
        out[f"b{b}_res"] = norm_res[b]
        out[f"b{b}_innov"] = innov[b]
        out[f"b{b}_nb_mean_innov"] = float(np.mean(nb_innov))
        out[f"b{b}_nb_max_innov"] = float(np.max(nb_innov))
        out[f"b{b}_nb_mean_res"] = float(np.mean(nb_res))
        out[f"b{b}_nb_max_res"] = float(np.max(nb_res))
        out[f"b{b}_local_minus_nb_innov"] = float(
            innov[b] - np.mean(nb_innov)
        )
    return out


def main():
    rng = np.random.default_rng(RANDOM_SEED)
    slow = pd.read_csv(SLOW_PATH)

    print("\n=== PHASE 2T: END-TO-END DECISION LATENCY ===")
    print(f"Protection-grade budget (assumed 50 Hz) : {CYCLE_MS:.2f} ms / cycle")
    print(f"Trials                                    : {N_TRIALS}")

    # ---- One-time setup: NOT part of per-decision latency ----
    net = topology.build_microgrid()
    phase21.configure_context(net, slow.iloc[0])
    phase21.solve_truth(net)
    model = phase21.build_measurement_model(net)
    names, stds = phase21.measurement_schema(net)
    adjacency = build_adjacency(net)
    node_ids = sorted(int(b) for b in net.bus.index)

    load_buses = [
        int(net.gridra["bus_load_a"]),
        int(net.gridra["bus_load_b"]),
    ]

    # Quick classifier trained on a handful of synthetic feature
    # vectors of the SAME shape/dtype as timed_feature_step() produces,
    # purely so .predict() below is timing a real fitted sklearn model
    # of realistic size, not a stub.
    n_feat = len(timed_feature_step(node_ids, adjacency,
                                     np.zeros(5), np.zeros(5)))
    X_dummy = rng.normal(size=(200, n_feat))
    y_dummy = rng.integers(0, 2, size=200)
    clf = Pipeline([
        ("impute", SimpleImputer()),
        ("scale", StandardScaler()),
        ("model", RandomForestClassifier(
            n_estimators=200, random_state=RANDOM_SEED
        )),
    ])
    clf.fit(X_dummy, y_dummy)

    wls_ms, feat_ms, pred_ms, total_ms = [], [], [], []

    for i in range(N_TRIALS):
        target_bus = int(rng.choice(load_buses))
        context = slow.iloc[int(rng.integers(0, len(slow)))]

        vm_true, va_true = phase21.truth_state(net)
        h_true = phase21.h_ac(model, vm_true, va_true)
        noise = rng.normal(0.0, 1.0, size=len(stds))
        z_clean = h_true + stds * noise

        z_used, _, _ = phase21.apply_model_consistent_attack(
            z_clean, h_true, model, vm_true, va_true, target_bus, rng
        )

        # ---- Stage 1: WLS state estimation ----
        t0 = time.perf_counter()
        result = phase21.run_wls(net, model, z_used, stds)
        t1 = time.perf_counter()

        if result is None:
            continue

        vm_hat = result["vm_hat"]
        va_hat = result["va_hat"]
        norm_res = np.abs(vm_hat - vm_true) / (stds[: len(vm_hat)] + 1e-9)
        innov = np.abs(va_hat - va_true)

        # ---- Stage 2: feature computation ----
        t2 = time.perf_counter()
        feats = timed_feature_step(node_ids, adjacency, norm_res, innov)
        t3 = time.perf_counter()

        # ---- Stage 3: model inference ----
        X = np.array([[feats[k] for k in sorted(feats)]])
        t4 = time.perf_counter()
        _ = clf.predict(X)
        t5 = time.perf_counter()

        wls_ms.append((t1 - t0) * 1000.0)
        feat_ms.append((t3 - t2) * 1000.0)
        pred_ms.append((t5 - t4) * 1000.0)
        total_ms.append(((t1 - t0) + (t3 - t2) + (t5 - t4)) * 1000.0)

    def stats(x):
        a = np.array(x)
        return {
            "median_ms": float(np.median(a)),
            "p95_ms": float(np.percentile(a, 95)),
            "max_ms": float(np.max(a)),
        }

    summary = pd.DataFrame({
        "stage": ["wls_state_estimation", "feature_computation",
                  "model_inference", "TOTAL"],
        **{
            k: [stats(wls_ms)[k], stats(feat_ms)[k],
                stats(pred_ms)[k], stats(total_ms)[k]]
            for k in ["median_ms", "p95_ms", "max_ms"]
        },
    })

    print(f"\nSuccessful trials: {len(total_ms)} / {N_TRIALS}\n")
    print(summary.to_string(index=False))

    over_budget = float(np.median(total_ms)) > CYCLE_MS
    print(
        f"\nMedian total latency {np.median(total_ms):.3f} ms vs "
        f"{CYCLE_MS:.2f} ms budget -> "
        f"{'OVER BUDGET' if over_budget else 'within budget'}"
    )

    RESULTS.mkdir(exist_ok=True)
    summary.to_csv(
        RESULTS / "phase2t_latency_benchmark.csv", index=False
    )
    print("\nSaved:\n  results/phase2t_latency_benchmark.csv")


if __name__ == "__main__":
    main()
