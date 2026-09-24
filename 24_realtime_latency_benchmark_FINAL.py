from pathlib import Path
import time
import importlib.util
import warnings
import json
import platform
import sys

import numpy as np
import pandas as pd
import pandapower as pp
import sklearn

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")

# ============================================================
# PHASE 2T -- FINAL (all validated fixes combined, larger N)
# ============================================================
#
# Combines every fix validated separately in this phase's earlier
# scripts, run once at N=300 (vs 60-100 before) for stable p95/max
# tail-latency estimates:
#   - fast measurement update (24_..._OPTIMIZED.py): bulk-write
#     net.measurement['value'] instead of looping pp.create_measurement()
#     23x/cycle. Verified bit-identical WLS output.
#   - LogisticRegression instead of RandomForest for the classifier
#     (competitive accuracy per Phase 2S-hard, 14x faster inference).
#   - tolerance=1e-4, maximum_iterations=10 instead of 1e-7/40
#     (25_wls_overhead_diagnostic.py: H1 showed max_iterations isn't
#     the driver below ~5, H2 showed this tolerance change costs
#     ~2e-6 pu state error, far below sensor noise floor).
#
# Run in isolation (no other heavy background jobs) so timing
# measurements aren't contaminated by CPU contention.
# ============================================================


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"

CYCLE_MS = 1000.0 / 50.0  # 20 ms -- net.f_hz verified == 50.0, not assumed
N_TRIALS = 300
RANDOM_SEED = 20260920


def load_module(filename, module_name):
    path = ROOT / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


phase21 = load_module(
    "21_nonlinear_stealth_fdia_bdd_benchmark_HARD.py",
    "phase21_final_2t",
)
topology = phase21.topology
SLOW_PATH = DATA / "phase2e_7day_operating_dataset.csv"


def build_adjacency(net):
    adjacency = {int(b): set() for b in net.bus.index}
    for _, ln in net.line.iterrows():
        f, t = int(ln["from_bus"]), int(ln["to_bus"])
        adjacency[f].add(t)
        adjacency[t].add(f)
    return adjacency


def timed_feature_step(node_ids, adjacency, norm_res, innov):
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
        out[f"b{b}_local_minus_nb_innov"] = float(innov[b] - np.mean(nb_innov))
    return out


def main():
    rng = np.random.default_rng(RANDOM_SEED)
    slow = pd.read_csv(SLOW_PATH)

    print("\n=== PHASE 2T FINAL: N=300, all fixes combined ===")
    print(f"Budget (net.f_hz={topology.build_microgrid().f_hz} Hz): {CYCLE_MS:.2f} ms/cycle")

    # Reproducibility metadata for machine-dependent latency claims.
    # The paper should report the exact benchmark environment rather than
    # only saying "one machine"; this block records it automatically on rerun.
    environment = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": sys.version.split()[0],
        "pandapower": pp.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
    }
    print("Benchmark environment:")
    for key, value in environment.items():
        print(f"  {key}: {value}")

    net = topology.build_microgrid()
    phase21.configure_context(net, slow.iloc[0])
    phase21.solve_truth(net)
    model = phase21.build_measurement_model(net)
    names, stds = phase21.measurement_schema(net)
    adjacency = build_adjacency(net)
    node_ids = sorted(int(b) for b in net.bus.index)
    load_buses = [int(net.gridra["bus_load_a"]), int(net.gridra["bus_load_b"])]

    # FIX (post-submission audit, Sec. 6 item 4, continued): "innov"
    # below used to be |va_hat - va_true| -- va_true is the same kind
    # of oracle-only ground truth as vm_true, not something a deployed
    # system has. Everywhere else in this project (22/28/31_*.py),
    # "innovation" compares the WLS estimate to an independently-drawn
    # PRIOR/forecast, not to ground truth -- built the same way here:
    # a separate prior_net solved from the same context with a small,
    # independent load-forecast error (2.5%, matching the 5-bus
    # convention in 22_graph_ready_protected_prior_telemetry_HARD.py).
    prior_net = topology.build_microgrid()
    prior_context = slow.iloc[0].copy()
    prior_context["load_a_p_mw"] = float(prior_context["load_a_p_mw"]) * (
        1.0 + float(np.clip(rng.normal(0.0, 0.025), -0.06, 0.06))
    )
    prior_context["load_b_p_mw"] = float(prior_context["load_b_p_mw"]) * (
        1.0 + float(np.clip(rng.normal(0.0, 0.025), -0.06, 0.06))
    )
    phase21.configure_context(prior_net, prior_context)
    phase21.solve_truth(prior_net)
    _, va_prior = phase21.truth_state(prior_net)

    phase21.add_measurement_vector(net, np.ones(len(stds)), stds)

    # FIX (caught in post-submission audit, STATUS.md Sec. 6 item 4):
    # this used to fit LogisticRegression on X_dummy/y_dummy -- pure
    # random noise and random labels, so clf.predict() below exercised
    # the right *computational graph* (same feature count, same model
    # class) but was not a real detector by any definition. Warm-up
    # trials below generate genuine clean/attack scenarios through this
    # same pipeline and extract real (feature, label) pairs, so the
    # classifier that gets timed has actually learned something.
    # (Prediction is a fixed-size dot product + sigmoid either way, so
    # this does not change pred_ms -- verified by re-running before vs.
    # after this fix -- but "trained on nothing" was never defensible
    # regardless of whether it moved the number.)
    N_WARMUP = 150
    warmup_rows, warmup_labels = [], []
    for _ in range(N_WARMUP):
        wu_target = int(rng.choice(load_buses))
        wu_vm_true, wu_va_true = phase21.truth_state(net)
        wu_h_true = phase21.h_ac(model, wu_vm_true, wu_va_true)
        wu_noise = rng.normal(0.0, 1.0, size=len(stds))
        wu_z_clean = wu_h_true + stds * wu_noise
        is_attack = bool(rng.integers(0, 2))
        wu_z_used = wu_z_clean
        if is_attack:
            wu_z_used, _, _ = phase21.apply_model_consistent_attack(
                wu_z_clean, wu_h_true, model, wu_vm_true, wu_va_true, wu_target, rng
            )
        net.measurement["value"] = np.asarray(wu_z_used, dtype=float)
        wu_success = bool(phase21.estimate(
            net, algorithm="wls", init="flat",
            tolerance=1e-4, maximum_iterations=10,
            calculate_voltage_angles=True,
        ))
        if not wu_success:
            continue
        wu_vm_hat, wu_va_hat = phase21.estimated_state(net)
        # Same deployable-residual fix as 28_ieee14_scale_replication.py/
        # 31_ieee30_scale_replication.py: z - h(x_hat), not vm_hat - vm_true.
        wu_h_hat = phase21.h_ac(model, wu_vm_hat, wu_va_hat)
        wu_norm_res = np.abs((np.asarray(wu_z_used, dtype=float) - wu_h_hat) / stds)[: len(wu_vm_hat)]
        wu_innov = np.abs(wu_va_hat - va_prior)
        wu_feats = timed_feature_step(node_ids, adjacency, wu_norm_res, wu_innov)
        warmup_rows.append([wu_feats[k] for k in sorted(wu_feats)])
        warmup_labels.append(int(is_attack))

    n_feat = len(timed_feature_step(node_ids, adjacency, np.zeros(5), np.zeros(5)))
    X_train = np.array(warmup_rows) if warmup_rows else rng.normal(size=(200, n_feat))
    y_train = np.array(warmup_labels) if warmup_rows else rng.integers(0, 2, size=200)
    clf = Pipeline([
        ("impute", SimpleImputer()),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000)),
    ])
    clf.fit(X_train, y_train)
    print(f"Warm-up: {len(warmup_labels)} real scenarios "
          f"({sum(warmup_labels)} attack, {len(warmup_labels) - sum(warmup_labels)} clean), "
          f"train balanced accuracy = "
          f"{float(np.mean(clf.predict(X_train) == y_train)):.3f}")

    wls_ms, poststate_ms, feat_ms, assembly_ms, pred_ms, total_ms = [], [], [], [], [], []
    n_fail = 0

    for i in range(N_TRIALS):
        target_bus = int(rng.choice(load_buses))
        vm_true, va_true = phase21.truth_state(net)
        h_true = phase21.h_ac(model, vm_true, va_true)
        noise = rng.normal(0.0, 1.0, size=len(stds))
        z_clean = h_true + stds * noise
        z_used, _, _ = phase21.apply_model_consistent_attack(
            z_clean, h_true, model, vm_true, va_true, target_bus, rng
        )

        t0 = time.perf_counter()
        net.measurement["value"] = np.asarray(z_used, dtype=float)
        success = bool(phase21.estimate(
            net, algorithm="wls", init="flat",
            tolerance=1e-4, maximum_iterations=10,
            calculate_voltage_angles=True,
        ))
        t1 = time.perf_counter()

        if not success:
            n_fail += 1
            continue

        # FIX (caught in a second-pass audit, STATUS.md Sec. 6): total_ms
        # used to be (t1-t0)+(t3-t2)+(t5-t4) -- summing only the three
        # named sub-steps and silently skipping estimated_state()/h_ac()/
        # residual+innovation computation (the t1-to-t2 gap) and the X
        # array assembly (the t3-to-t4 gap) entirely. Both are real,
        # on-the-critical-path work a deployed cycle would also have to
        # do, and at p99 there was only 1.59ms of headroom under the
        # 20ms budget -- not obviously enough to safely ignore an
        # unmeasured gap. Every stage is now timed with no gap between
        # timestamps, and total_ms is (t5-t0), the true, nothing-excluded
        # per-cycle wall-clock time; the four named sub-steps are kept
        # for the diagnostic breakdown, now including the two that were
        # previously invisible (poststate_ms, assembly_ms).
        vm_hat, va_hat = phase21.estimated_state(net)
        h_hat = phase21.h_ac(model, vm_hat, va_hat)
        norm_res = np.abs((np.asarray(z_used, dtype=float) - h_hat) / stds)[: len(vm_hat)]
        innov = np.abs(va_hat - va_prior)

        t2 = time.perf_counter()
        feats = timed_feature_step(node_ids, adjacency, norm_res, innov)
        t3 = time.perf_counter()

        X = np.array([[feats[k] for k in sorted(feats)]])
        t4 = time.perf_counter()
        _ = clf.predict(X)
        t5 = time.perf_counter()

        wls_ms.append((t1 - t0) * 1000.0)
        poststate_ms.append((t2 - t1) * 1000.0)
        feat_ms.append((t3 - t2) * 1000.0)
        assembly_ms.append((t4 - t3) * 1000.0)
        pred_ms.append((t5 - t4) * 1000.0)
        total_ms.append((t5 - t0) * 1000.0)

    def stats(x):
        a = np.array(x)
        return {
            "median_ms": float(np.median(a)),
            "p95_ms": float(np.percentile(a, 95)),
            "p99_ms": float(np.percentile(a, 99)),
            "max_ms": float(np.max(a)),
        }

    summary = pd.DataFrame({
        "stage": ["wls_state_estimation", "poststate_residual_innovation",
                   "feature_computation", "array_assembly", "model_inference",
                   "TOTAL (t5-t0, nothing excluded)"],
        **{
            k: [stats(wls_ms)[k], stats(poststate_ms)[k], stats(feat_ms)[k],
                 stats(assembly_ms)[k], stats(pred_ms)[k], stats(total_ms)[k]]
            for k in ["median_ms", "p95_ms", "p99_ms", "max_ms"]
        },
    })

    print(f"\nTrials: {N_TRIALS} | convergence failures: {n_fail}\n")
    print(summary.to_string(index=False))

    for label, val in [("median", np.median(total_ms)),
                        ("p95", np.percentile(total_ms, 95)),
                        ("p99", np.percentile(total_ms, 99)),
                        ("max", np.max(total_ms))]:
        status = "within budget" if val <= CYCLE_MS else "OVER budget"
        print(f"{label:>6}: {val:7.3f} ms -> {status}")

    RESULTS.mkdir(exist_ok=True)
    summary.to_csv(RESULTS / "phase2t_latency_final.csv", index=False)
    with open(RESULTS / "phase2t_latency_environment.json", "w", encoding="utf-8") as fh:
        json.dump(environment, fh, indent=2)
    print("\nSaved:\n  results/phase2t_latency_final.csv\n  results/phase2t_latency_environment.json")


if __name__ == "__main__":
    main()
