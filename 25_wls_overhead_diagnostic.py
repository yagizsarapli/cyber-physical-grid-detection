from pathlib import Path
import time
import importlib.util
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ============================================================
# PHASE 2T -- WLS OVERHEAD DIAGNOSTIC
# ============================================================
#
# Warm-start only bought 1.06x (24_realtime_latency_benchmark_
# WARMSTART.py), which rules out "far from the solution needs many
# Newton iterations" as the dominant cost. This script isolates
# WHICH knob actually matters by testing two hypotheses directly:
#
#   H1: maximum_iterations=40 is mostly unused headroom -- the
#       solver already converges in far fewer steps, so iteration
#       count was never the bottleneck (points at fixed per-call
#       pandapower overhead instead).
#   H2: tolerance=1e-7 forces extra iterations beyond what sensor
#       noise justifies -- loosening it should measurably help.
#
# Method: same single scenario, re-run estimate() with varying
# maximum_iterations (tolerance fixed) and varying tolerance
# (maximum_iterations fixed), timing each and recording whether it
# still converges and how far its state is from the tight-tolerance
# answer.
# ============================================================


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"

N_TRIALS = 20
RANDOM_SEED = 20260920


def load_module(filename, module_name):
    path = ROOT / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


phase21 = load_module(
    "21_nonlinear_stealth_fdia_bdd_benchmark_HARD.py",
    "phase21_hard_for_diag",
)
topology = phase21.topology
SLOW_PATH = DATA / "phase2e_7day_operating_dataset.csv"


def one_estimate(net, model, z, stds, max_iter, tol):
    phase21.add_measurement_vector(net, z, stds)
    t0 = time.perf_counter()
    success = bool(
        phase21.estimate(
            net, algorithm="wls", init="flat",
            tolerance=tol, maximum_iterations=max_iter,
            calculate_voltage_angles=True,
        )
    )
    t1 = time.perf_counter()
    if not success:
        return None, (t1 - t0) * 1000.0
    vm_hat, va_hat = phase21.estimated_state(net)
    return (vm_hat, va_hat), (t1 - t0) * 1000.0


def main():
    rng = np.random.default_rng(RANDOM_SEED)
    slow = pd.read_csv(SLOW_PATH)

    net = topology.build_microgrid()
    phase21.configure_context(net, slow.iloc[0])
    phase21.solve_truth(net)
    model = phase21.build_measurement_model(net)
    names, stds = phase21.measurement_schema(net)
    load_buses = [int(net.gridra["bus_load_a"]), int(net.gridra["bus_load_b"])]

    # Fixed pool of measurement vectors reused across every config
    # so only max_iter/tolerance differ, not the underlying problem.
    scenarios = []
    for _ in range(N_TRIALS):
        target_bus = int(rng.choice(load_buses))
        vm_true, va_true = phase21.truth_state(net)
        h_true = phase21.h_ac(model, vm_true, va_true)
        noise = rng.normal(0.0, 1.0, size=len(stds))
        z_clean = h_true + stds * noise
        z_used, _, _ = phase21.apply_model_consistent_attack(
            z_clean, h_true, model, vm_true, va_true, target_bus, rng
        )
        scenarios.append((z_used, vm_true, va_true))

    print("\n=== H1: does maximum_iterations actually matter? (tolerance=1e-7) ===")
    for max_iter in [40, 20, 10, 5, 3, 2, 1]:
        times, n_ok = [], 0
        for z_used, vm_true, va_true in scenarios:
            res, ms = one_estimate(net, model, z_used, stds, max_iter, 1e-7)
            times.append(ms)
            n_ok += res is not None
        print(f"  max_iter={max_iter:>3} | converged {n_ok:>2}/{N_TRIALS} | "
              f"median {np.median(times):7.3f} ms | mean {np.mean(times):7.3f} ms")

    print("\n=== H2: does loosening tolerance matter? (maximum_iterations=40) ===")
    baseline_states = []
    for z_used, vm_true, va_true in scenarios:
        res, _ = one_estimate(net, model, z_used, stds, 40, 1e-7)
        baseline_states.append(res)

    for tol in [1e-7, 1e-5, 1e-4, 1e-3, 1e-2]:
        times, n_ok, state_err = [], 0, []
        for (z_used, vm_true, va_true), base in zip(scenarios, baseline_states):
            res, ms = one_estimate(net, model, z_used, stds, 40, tol)
            times.append(ms)
            if res is not None:
                n_ok += 1
                if base is not None:
                    d = np.max(np.abs(res[0] - base[0]))
                    state_err.append(d)
        err_str = f"{np.max(state_err):.2e}" if state_err else "n/a"
        print(f"  tol={tol:.0e} | converged {n_ok:>2}/{N_TRIALS} | "
              f"median {np.median(times):7.3f} ms | mean {np.mean(times):7.3f} ms | "
              f"max |dVm| vs tol=1e-7 baseline: {err_str} pu")


if __name__ == "__main__":
    main()
