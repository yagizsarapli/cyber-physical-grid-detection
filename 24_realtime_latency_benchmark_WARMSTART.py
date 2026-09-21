from pathlib import Path
import time
import importlib.util
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ============================================================
# PHASE 2T -- WARM-START VARIANT
# ============================================================
#
# 24_realtime_latency_benchmark.py found the WLS solve alone costs
# ~40 ms median (flat init="flat" every cycle), ~93% of total
# end-to-end latency, 2x over the 20 ms/cycle protection budget.
#
# This script tests the obvious next lever: warm-starting each
# cycle's WLS solve from the PREVIOUS cycle's converged state
# (init="results") instead of flat-starting (1.0 pu / 0 deg) every
# time. This mirrors real deployment: within one sustained event
# (a fault or an attack usually persists for many cycles), the true
# state barely moves between consecutive cycles while the sensors
# report a fresh independent noisy sample each cycle.
#
# Design: for each of N_EVENTS events, run K_CYCLES consecutive
# cycles sharing the SAME true state/attack but fresh independent
# noise per cycle. Cycle 0 has no prior estimate -> flat init
# ("cold"). Cycles 1..K-1 warm-start from the immediately preceding
# cycle's own estimate ("warm"). This is a fair test: it never
# re-solves the identical measurement vector twice.
#
# We also check warm-start doesn't silently degrade accuracy
# (compare final WLS residual/J-statistic between cold and warm).
# ============================================================


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"

CYCLE_MS = 1000.0 / 50.0  # 20 ms, 50 Hz assumption
N_EVENTS = 30
K_CYCLES = 5
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
    "phase21_hard_for_2t_warm",
)
topology = phase21.topology

SLOW_PATH = DATA / "phase2e_7day_operating_dataset.csv"


def run_wls_variant(net, model, z, stds, init):
    """Same body as phase21.run_wls, but with a selectable init mode
    so we can compare 'flat' vs 'results' (warm-start)."""
    phase21.add_measurement_vector(net, z, stds)

    success = bool(
        phase21.estimate(
            net,
            algorithm="wls",
            init=init,
            tolerance=1e-7,
            maximum_iterations=40,
            calculate_voltage_angles=True,
        )
    )
    if not success:
        return None

    vm_hat, va_hat = phase21.estimated_state(net)
    h_hat = phase21.h_ac(model, vm_hat, va_hat)
    residual = np.asarray(z, dtype=float) - h_hat
    norm_residual = residual / stds
    j_stat = float(np.sum(norm_residual ** 2))

    return {"vm_hat": vm_hat, "va_hat": va_hat, "j_stat": j_stat}


def main():
    rng = np.random.default_rng(RANDOM_SEED)
    slow = pd.read_csv(SLOW_PATH)

    print("\n=== PHASE 2T -- WARM-START WLS ===")
    print(f"Budget                 : {CYCLE_MS:.2f} ms/cycle (50 Hz assumption)")
    print(f"Events x cycles/event  : {N_EVENTS} x {K_CYCLES}")

    net = topology.build_microgrid()
    phase21.configure_context(net, slow.iloc[0])
    phase21.solve_truth(net)
    model = phase21.build_measurement_model(net)
    names, stds = phase21.measurement_schema(net)

    load_buses = [
        int(net.gridra["bus_load_a"]),
        int(net.gridra["bus_load_b"]),
    ]

    cold_ms, warm_ms = [], []
    cold_j, warm_j = [], []
    warm_failures = 0

    for e in range(N_EVENTS):
        target_bus = int(rng.choice(load_buses))
        context = slow.iloc[int(rng.integers(0, len(slow)))]

        # Same true state for the whole event (a sustained attack),
        # fresh measurement noise drawn each cycle below.
        vm_true, va_true = phase21.truth_state(net)
        h_true = phase21.h_ac(model, vm_true, va_true)

        for c in range(K_CYCLES):
            noise = rng.normal(0.0, 1.0, size=len(stds))
            z_clean = h_true + stds * noise
            z_used, _, _ = phase21.apply_model_consistent_attack(
                z_clean, h_true, model, vm_true, va_true, target_bus, rng
            )

            init = "flat" if c == 0 else "results"

            t0 = time.perf_counter()
            result = run_wls_variant(net, model, z_used, stds, init)
            t1 = time.perf_counter()

            if result is None:
                if c > 0:
                    warm_failures += 1
                continue

            elapsed_ms = (t1 - t0) * 1000.0
            if c == 0:
                cold_ms.append(elapsed_ms)
                cold_j.append(result["j_stat"])
            else:
                warm_ms.append(elapsed_ms)
                warm_j.append(result["j_stat"])

    def stats(x):
        a = np.array(x)
        return float(np.median(a)), float(np.percentile(a, 95)), float(np.max(a))

    cold_med, cold_p95, cold_max = stats(cold_ms)
    warm_med, warm_p95, warm_max = stats(warm_ms)

    print(f"\nCold (flat init, cycle 0 of each event), n={len(cold_ms)}:")
    print(f"  median {cold_med:.3f} ms | p95 {cold_p95:.3f} ms | max {cold_max:.3f} ms")
    print(f"\nWarm (results init, cycles 1..{K_CYCLES-1}), n={len(warm_ms)}"
          f" (failed to converge: {warm_failures}):")
    print(f"  median {warm_med:.3f} ms | p95 {warm_p95:.3f} ms | max {warm_max:.3f} ms")

    print(f"\nAccuracy check (WLS J-statistic, lower = better fit):")
    print(f"  cold median J = {np.median(cold_j):.4f}")
    print(f"  warm median J = {np.median(warm_j):.4f}")

    speedup = cold_med / warm_med if warm_med > 0 else float("nan")
    print(f"\nWarm-start speedup vs cold: {speedup:.2f}x")
    print(
        f"Warm median {warm_med:.3f} ms vs {CYCLE_MS:.2f} ms budget -> "
        f"{'OVER BUDGET' if warm_med > CYCLE_MS else 'WITHIN BUDGET'}"
    )

    RESULTS.mkdir(exist_ok=True)
    pd.DataFrame({
        "condition": ["cold_flat", "warm_results"],
        "n": [len(cold_ms), len(warm_ms)],
        "median_ms": [cold_med, warm_med],
        "p95_ms": [cold_p95, warm_p95],
        "max_ms": [cold_max, warm_max],
        "median_j_stat": [float(np.median(cold_j)), float(np.median(warm_j))],
    }).to_csv(RESULTS / "phase2t_warmstart_comparison.csv", index=False)
    print("\nSaved:\n  results/phase2t_warmstart_comparison.csv")


if __name__ == "__main__":
    main()
