from pathlib import Path
import importlib.util
import copy
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandapower as pp

from pandapower.estimation import estimate

logging.getLogger("pandapower").setLevel(logging.CRITICAL)

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)


def load_module(filename, module_name):
    spec = importlib.util.spec_from_file_location(module_name, ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


topology = load_module("01_microgrid_topology.py", "topology")

# ============================================================
# PHASE 2D — ROBUST STATE ESTIMATION + LOCALIZATION BENCHMARK
# ============================================================

RNG = np.random.default_rng(20260811)

# Synthetic SCADA uncertainty assumptions.
STD_V_PU = 0.005
STD_P_MW = 0.0010
STD_Q_MVAR = 0.0008
STD_LINE_P_MW = 0.0012
STD_LINE_Q_MVAR = 0.0010

# Residual threshold used only as a common comparison metric.
RESIDUAL_THRESHOLD = 3.0

ESTIMATORS = [
    {
        "name": "WLS",
        "warm_start": False,
        "kwargs": {
            "algorithm": "wls",
            "init": "flat",
            "tolerance": 1e-7,
            "maximum_iterations": 30,
            "calculate_voltage_angles": True,
        },
    },
    {
        "name": "SHGM",
        # pandapower 2.14.10 on the user's environment:
        # SHGM does not reliably converge from flat start, but does converge
        # when initialized from a converged WLS state.
        "warm_start": True,
        "kwargs": {
            "algorithm": "irwls",
            "estimator": "shgm",
            "a": 5,
            "init": "results",
            "tolerance": 1e-7,
            "maximum_iterations": 50,
            "calculate_voltage_angles": True,
        },
    },
    {
        "name": "LAV",
        "warm_start": False,
        "kwargs": {
            "algorithm": "lp",
            "init": "flat",
            "tolerance": 1e-7,
            "maximum_iterations": 50,
            "calculate_voltage_angles": True,
        },
    },
]


# ------------------------------------------------------------
# Benchmark cases
# ------------------------------------------------------------
# The benchmark intentionally contains:
#   1) clean noise-only control
#   2) real physical load change
#   3) voltage-sensor bias sweep
#   4) line-flow sensor bias sweep
#   5) mild coordinated multi-sensor corruption
#
# These are defensive robustness tests. No residual-evasion
# optimization is performed in this phase.
CASES = [
    {
        "case": "clean_noisy",
        "family": "normal",
        "severity": 0.0,
        "unit": "",
    },
    {
        "case": "physical_load_step",
        "family": "physical",
        "severity": 0.0,
        "unit": "",
    },
]

for bias in [0.010, 0.020, 0.040, 0.060, 0.080]:
    CASES.append({
        "case": "voltage_bias",
        "family": "corruption",
        "severity": bias,
        "unit": "pu",
    })

for bias in [0.005, 0.010, 0.015, 0.025]:
    CASES.append({
        "case": "line_flow_bias",
        "family": "corruption",
        "severity": bias,
        "unit": "MW",
    })

CASES.append({
    "case": "coordinated_mild",
    "family": "corruption",
    "severity": 1.0,
    "unit": "case",
})


def build_truth_network(case_name):
    net = topology.build_microgrid()

    if case_name == "physical_load_step":
        idx = net.load.index[0]
        net.load.at[idx, "p_mw"] = 0.055
        net.load.at[idx, "q_mvar"] = 0.014

    topology.solve_base_case(net)
    return net


def noisy(value, std):
    return float(value + RNG.normal(0.0, std))


def add_measurements_from_truth(net):
    if len(net.measurement):
        net.measurement.drop(net.measurement.index, inplace=True)

    for bus in net.bus.index:
        pp.create_measurement(
            net, "v", "bus",
            noisy(float(net.res_bus.at[bus, "vm_pu"]), STD_V_PU),
            STD_V_PU, int(bus),
            name=f"V_bus_{bus}",
        )

        pp.create_measurement(
            net, "p", "bus",
            noisy(float(net.res_bus.at[bus, "p_mw"]), STD_P_MW),
            STD_P_MW, int(bus),
            name=f"P_bus_{bus}",
        )

        pp.create_measurement(
            net, "q", "bus",
            noisy(float(net.res_bus.at[bus, "q_mvar"]), STD_Q_MVAR),
            STD_Q_MVAR, int(bus),
            name=f"Q_bus_{bus}",
        )

    for line in net.line.index:
        pp.create_measurement(
            net, "p", "line",
            noisy(float(net.res_line.at[line, "p_from_mw"]), STD_LINE_P_MW),
            STD_LINE_P_MW, int(line),
            side="from",
            name=f"P_line_{line}_from",
        )

        pp.create_measurement(
            net, "q", "line",
            noisy(float(net.res_line.at[line, "q_from_mvar"]), STD_LINE_Q_MVAR),
            STD_LINE_Q_MVAR, int(line),
            side="from",
            name=f"Q_line_{line}_from",
        )


def corrupt_by_name(net, name, delta):
    mask = net.measurement["name"] == name
    if not mask.any():
        raise KeyError(name)

    idx = net.measurement.index[mask][0]
    net.measurement.at[idx, "value"] += float(delta)
    return int(idx)


def apply_case_corruption(net, case):
    corrupted = []

    if case["case"] == "voltage_bias":
        corrupted.append(
            corrupt_by_name(
                net,
                "V_bus_4",
                case["severity"],
            )
        )

    elif case["case"] == "line_flow_bias":
        corrupted.append(
            corrupt_by_name(
                net,
                "P_line_1_from",
                case["severity"],
            )
        )

    elif case["case"] == "coordinated_mild":
        # Multiple modest corruptions that are individually less dramatic
        # than the Phase 2C examples.
        corrupted += [
            corrupt_by_name(net, "V_bus_3", -0.015),
            corrupt_by_name(net, "V_bus_4", -0.020),
            corrupt_by_name(net, "P_bus_4", +0.006),
            corrupt_by_name(net, "Q_bus_4", +0.004),
            corrupt_by_name(net, "P_line_3_from", +0.007),
        ]

    return corrupted


def estimated_measurement_value(net, m):
    mt = m["measurement_type"]
    et = m["element_type"]
    element = int(m["element"])

    if et == "bus":
        if mt == "v":
            return float(net.res_bus_est.at[element, "vm_pu"])
        if mt == "p":
            return float(net.res_bus_est.at[element, "p_mw"])
        if mt == "q":
            return float(net.res_bus_est.at[element, "q_mvar"])

    if et == "line":
        side = m["side"]
        if side not in ("from", "to"):
            side = "from"

        if mt == "p":
            col = "p_from_mw" if side == "from" else "p_to_mw"
            return float(net.res_line_est.at[element, col])

        if mt == "q":
            col = "q_from_mvar" if side == "from" else "q_to_mvar"
            return float(net.res_line_est.at[element, col])

    return np.nan


def residual_table(net, corrupted_indices):
    rows = []

    for idx, m in net.measurement.iterrows():
        est_val = estimated_measurement_value(net, m)

        if not np.isfinite(est_val):
            continue

        measured = float(m["value"])
        std = float(m["std_dev"])

        residual = measured - est_val
        nr = abs(residual) / max(std, 1e-12)

        rows.append({
            "measurement_index": int(idx),
            "name": str(m["name"]),
            "measured": measured,
            "estimated": est_val,
            "std_dev": std,
            "residual": residual,
            "abs_normalized_residual": nr,
            "is_corrupted": int(idx) in corrupted_indices,
        })

    return pd.DataFrame(rows)


def state_metrics(net):
    v_true = net.res_bus["vm_pu"].to_numpy(dtype=float)
    v_est = net.res_bus_est["vm_pu"].to_numpy(dtype=float)

    a_true = net.res_bus["va_degree"].to_numpy(dtype=float)
    a_est = net.res_bus_est["va_degree"].to_numpy(dtype=float)

    return (
        float(np.sqrt(np.mean((v_est - v_true)**2))),
        float(np.sqrt(np.mean((a_est - a_true)**2))),
    )


def rank_localization(residuals, corrupted_indices):
    if residuals.empty or not corrupted_indices:
        return np.nan, np.nan, np.nan

    ranked = residuals.sort_values(
        "abs_normalized_residual",
        ascending=False,
    )

    top1 = set(ranked.head(1)["measurement_index"].tolist())
    top3 = set(ranked.head(3)["measurement_index"].tolist())
    bad = set(corrupted_indices)

    top1_hit = len(top1 & bad) > 0
    top3_hit = len(top3 & bad) > 0

    # Mean rank of corrupted measurements.
    rank_map = {
        idx: rank + 1
        for rank, idx in enumerate(
            ranked["measurement_index"].tolist()
        )
    }
    corrupted_ranks = [
        rank_map[idx]
        for idx in bad
        if idx in rank_map
    ]

    mean_rank = (
        float(np.mean(corrupted_ranks))
        if corrupted_ranks else np.nan
    )

    return bool(top1_hit), bool(top3_hit), mean_rank


def run_estimator(base_net, estimator, corrupted_indices):
    net = copy.deepcopy(base_net)

    try:
        # SHGM in pandapower 2.14.10 is sensitive to initialization.
        # Diagnostic test confirmed:
        #   flat start   -> fails
        #   WLS results -> converges
        if estimator.get("warm_start", False):
            wls_ok = bool(
                estimate(
                    net,
                    algorithm="wls",
                    init="flat",
                    tolerance=1e-7,
                    maximum_iterations=30,
                    calculate_voltage_angles=True,
                )
            )

            if not wls_ok:
                success = False
            else:
                success = bool(
                    estimate(
                        net,
                        **estimator["kwargs"],
                    )
                )
        else:
            success = bool(
                estimate(
                    net,
                    **estimator["kwargs"],
                )
            )

    except Exception:
        success = False

    if not success:
        return {
            "success": False,
            "v_rmse": np.nan,
            "angle_rmse": np.nan,
            "max_nr": np.nan,
            "detected": False,
            "top1_hit": np.nan,
            "top3_hit": np.nan,
            "mean_bad_rank": np.nan,
        }, pd.DataFrame()

    v_rmse, a_rmse = state_metrics(net)
    residuals = residual_table(net, corrupted_indices)

    max_nr = (
        float(residuals["abs_normalized_residual"].max())
        if not residuals.empty else np.nan
    )

    detected = (
        bool(max_nr >= RESIDUAL_THRESHOLD)
        if np.isfinite(max_nr) else False
    )

    top1, top3, mean_rank = rank_localization(
        residuals,
        corrupted_indices,
    )

    return {
        "success": True,
        "v_rmse": v_rmse,
        "angle_rmse": a_rmse,
        "max_nr": max_nr,
        "detected": detected,
        "top1_hit": top1,
        "top3_hit": top3,
        "mean_bad_rank": mean_rank,
    }, residuals

def run_one(case, repetition):
    truth = build_truth_network(case["case"])
    add_measurements_from_truth(truth)
    corrupted_indices = apply_case_corruption(truth, case)

    output_rows = []
    residual_examples = []

    for estimator in ESTIMATORS:
        metrics, residuals = run_estimator(
            truth,
            estimator,
            corrupted_indices,
        )

        output_rows.append({
            "case": case["case"],
            "family": case["family"],
            "severity": case["severity"],
            "severity_unit": case["unit"],
            "repetition": repetition,
            "estimator": estimator["name"],
            "n_measurements": len(truth.measurement),
            "n_corrupted": len(corrupted_indices),
            **metrics,
        })

        if repetition == 0 and not residuals.empty:
            r = residuals.copy()
            r["case"] = case["case"]
            r["severity"] = case["severity"]
            r["estimator"] = estimator["name"]
            residual_examples.append(r)

    return output_rows, residual_examples


def aggregate(df):
    grouped = df.groupby(
        ["case", "family", "severity", "severity_unit", "estimator"],
        as_index=False,
    ).agg(
        success_rate=("success", "mean"),
        detection_rate=("detected", "mean"),
        mean_v_rmse=("v_rmse", "mean"),
        p95_v_rmse=("v_rmse", lambda x: np.nanpercentile(x, 95)),
        mean_angle_rmse=("angle_rmse", "mean"),
        mean_max_nr=("max_nr", "mean"),
        top1_localization_rate=("top1_hit", "mean"),
        top3_localization_rate=("top3_hit", "mean"),
        mean_corrupted_rank=("mean_bad_rank", "mean"),
    )

    return grouped


def figure_voltage_detection_curve(agg):
    d = agg[agg["case"] == "voltage_bias"].copy()

    fig, ax = plt.subplots(figsize=(10, 6))

    for est in d["estimator"].unique():
        x = d[d["estimator"] == est]
        ax.plot(
            100*x["severity"],
            100*x["detection_rate"],
            marker="o",
            linewidth=2.0,
            label=est,
        )

    ax.set_xlabel("Voltage-measurement bias (% of 1 pu)")
    ax.set_ylabel("Detection rate (%)")
    ax.set_ylim(-3, 103)
    ax.set_title("Robust Estimator Detection Sensitivity to Voltage-Sensor Bias")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(
        FIGURES / "12_fixed_voltage_bias_detection_curve.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)


def figure_voltage_rmse_curve(agg):
    d = agg[agg["case"] == "voltage_bias"].copy()

    fig, ax = plt.subplots(figsize=(10, 6))

    for est in d["estimator"].unique():
        x = d[d["estimator"] == est]
        ax.plot(
            100*x["severity"],
            x["mean_v_rmse"],
            marker="o",
            linewidth=2.0,
            label=est,
        )

    ax.set_xlabel("Voltage-measurement bias (% of 1 pu)")
    ax.set_ylabel("Mean voltage-state RMSE (pu)")
    ax.set_title("State-Estimation Robustness Under Increasing Sensor Bias")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(
        FIGURES / "13_fixed_voltage_bias_state_rmse.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)


def figure_localization(agg):
    d = agg[
        agg["case"].isin(
            ["voltage_bias", "line_flow_bias", "coordinated_mild"]
        )
    ].copy()

    # Keep one representative severity for each family.
    keep = (
        ((d["case"] == "voltage_bias") & (d["severity"] == 0.040))
        | ((d["case"] == "line_flow_bias") & (d["severity"] == 0.015))
        | (d["case"] == "coordinated_mild")
    )
    d = d[keep].copy()

    d["label"] = d.apply(
        lambda r: (
            "4% voltage bias"
            if r["case"] == "voltage_bias"
            else "15 kW line-flow bias"
            if r["case"] == "line_flow_bias"
            else "Mild coordinated corruption"
        ),
        axis=1,
    )

    labels = list(dict.fromkeys(d["label"].tolist()))
    estimators = [e["name"] for e in ESTIMATORS]

    x = np.arange(len(labels))
    width = 0.24

    fig, ax = plt.subplots(figsize=(11, 6.2))

    for j, est in enumerate(estimators):
        vals = []
        for label in labels:
            row = d[
                (d["label"] == label)
                & (d["estimator"] == est)
            ]
            vals.append(
                100*float(row["top3_localization_rate"].iloc[0])
                if not row.empty else np.nan
            )

        ax.bar(
            x + (j-1)*width,
            vals,
            width,
            label=est,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Top-3 localization success (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Can the Estimator Residuals Localize the Corrupted Measurement?")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(
        FIGURES / "14_fixed_corruption_localization_benchmark.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)


def figure_false_alarm_control(agg):
    d = agg[
        agg["case"].isin(["clean_noisy", "physical_load_step"])
    ].copy()

    labels = ["Normal + noise", "Physical load step"]
    estimators = [e["name"] for e in ESTIMATORS]

    x = np.arange(len(labels))
    width = 0.24

    fig, ax = plt.subplots(figsize=(8.8, 5.8))

    for j, est in enumerate(estimators):
        vals = []
        for case_name in ["clean_noisy", "physical_load_step"]:
            row = d[
                (d["case"] == case_name)
                & (d["estimator"] == est)
            ]
            vals.append(
                100*float(row["detection_rate"].iloc[0])
                if not row.empty else np.nan
            )

        ax.bar(
            x + (j-1)*width,
            vals,
            width,
            label=est,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Residual-threshold flag rate (%)")
    ax.set_ylim(0, 105)
    ax.set_title("False-Alarm Control Under Legitimate Operating Conditions")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(
        FIGURES / "15_fixed_false_alarm_control.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)


if __name__ == "__main__":
    # 50 repetitions keeps runtime reasonable while producing a meaningful
    # first robustness map. Increase later for publication statistics.
    N_REPEATS = 50

    rows = []
    residual_frames = []

    print("\n=== PHASE 2D FIXED: ROBUST STATE ESTIMATION + LOCALIZATION ===")
    print(f"pandapower version: {pp.__version__}")
    print("SHGM initialization: WLS warm-start -> IRWLS-SHGM")
    print(f"Monte-Carlo repetitions per case: {N_REPEATS}")
    print("Estimators: WLS, SHGM, LAV")

    for case in CASES:
        print(
            f"\nRunning {case['case']} "
            f"(severity={case['severity']} {case['unit']}) ..."
        )

        case_rows = []

        for rep in range(N_REPEATS):
            run_rows, residuals = run_one(case, rep)
            rows.extend(run_rows)
            case_rows.extend(run_rows)
            residual_frames.extend(residuals)

        tmp = pd.DataFrame(case_rows)

        for est in tmp["estimator"].unique():
            e = tmp[tmp["estimator"] == est]

            print(
                f"  {est:4s} | "
                f"conv={100*e['success'].mean():5.1f}% | "
                f"flag={100*e['detected'].mean():5.1f}% | "
                f"V-RMSE={e['v_rmse'].mean():.5f} pu | "
                f"top3={100*e['top3_hit'].mean() if e['top3_hit'].notna().any() else np.nan:5.1f}%"
            )

    runs = pd.DataFrame(rows)
    agg = aggregate(runs)

    runs.to_csv(
        RESULTS / "phase2d_fixed_robust_estimator_runs.csv",
        index=False,
    )
    agg.to_csv(
        RESULTS / "phase2d_fixed_robust_estimator_summary.csv",
        index=False,
    )

    if residual_frames:
        residual_df = pd.concat(residual_frames, ignore_index=True)
        residual_df.to_csv(
            RESULTS / "phase2d_fixed_example_localization_residuals.csv",
            index=False,
        )

    figure_voltage_detection_curve(agg)
    figure_voltage_rmse_curve(agg)
    figure_localization(agg)
    figure_false_alarm_control(agg)

    print("\n=== AGGREGATED SUMMARY ===")
    print(
        agg[
            [
                "case",
                "severity",
                "estimator",
                "success_rate",
                "detection_rate",
                "mean_v_rmse",
                "top1_localization_rate",
                "top3_localization_rate",
            ]
        ].to_string(index=False)
    )

    print("\nSaved:")
    print("  results/phase2d_fixed_robust_estimator_runs.csv")
    print("  results/phase2d_fixed_robust_estimator_summary.csv")
    print("  results/phase2d_fixed_example_localization_residuals.csv")
    print("  figures/12_fixed_voltage_bias_detection_curve.png")
    print("  figures/13_fixed_voltage_bias_state_rmse.png")
    print("  figures/14_fixed_corruption_localization_benchmark.png")
    print("  figures/15_fixed_false_alarm_control.png")

    print(
        "\nNEXT RESEARCH STEP:\n"
        "Use the estimator outputs as a physics baseline, then add temporal "
        "and topology-aware features so detection/localization no longer "
        "depends on one residual threshold. The next model should explicitly "
        "combine grid graph structure, inverter internal states, state-"
        "estimation residuals and temporal history."
    )
