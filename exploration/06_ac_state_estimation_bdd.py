from pathlib import Path
import importlib.util
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandapower as pp

from pandapower.estimation import estimate, chi2_analysis

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
# PHASE 2C — AC STATE ESTIMATION + BAD-DATA BASELINE
# Defensive benchmark: noisy/redundant measurements, physical
# events, and non-stealth sensor corruptions.
# ============================================================

RNG = np.random.default_rng(42)

# Approximate sensor uncertainties for the synthetic benchmark.
STD_V_PU = 0.005          # 0.5 % voltage
STD_P_MW = 0.0010         # 1 kW
STD_Q_MVAR = 0.0008       # 0.8 kvar
STD_LINE_P_MW = 0.0012
STD_LINE_Q_MVAR = 0.0010

SCENARIOS = [
    {
        "name": "clean_noisy",
        "family": "normal",
        "description": "Normal operation with measurement noise only",
    },
    {
        "name": "physical_load_step",
        "family": "physical",
        "description": "Real load increase; measurements remain physically consistent",
    },
    {
        "name": "single_voltage_corruption",
        "family": "cyber",
        "description": "One bus-voltage measurement is corrupted",
    },
    {
        "name": "single_line_flow_corruption",
        "family": "cyber",
        "description": "One active line-flow measurement is corrupted",
    },
    {
        "name": "coordinated_measurement_corruption",
        "family": "cyber",
        "description": "Several related measurements are corrupted simultaneously",
    },
]


def build_truth_network(scenario_name):
    net = topology.build_microgrid()

    if scenario_name == "physical_load_step":
        # Real physical operating-point change.
        idx = net.load.index[0]
        net.load.at[idx, "p_mw"] = 0.055
        net.load.at[idx, "q_mvar"] = 0.014

    topology.solve_base_case(net)
    return net


def noisy(value, std):
    return float(value + RNG.normal(0.0, std))


def add_measurements_from_truth(net):
    """
    Build a redundant SCADA-like measurement set from the solved network.

    Measurement mix:
      - V at all 5 buses                         -> 5
      - P,Q injections at all 5 buses           -> 10
      - P,Q line flows at from-side of 4 lines  -> 8

    Total = 23 measurements for a 5-bus system.
    """
    if len(net.measurement):
        net.measurement.drop(net.measurement.index, inplace=True)

    # ---------- Bus voltage measurements ----------
    for bus in net.bus.index:
        v = float(net.res_bus.at[bus, "vm_pu"])
        pp.create_measurement(
            net,
            meas_type="v",
            element_type="bus",
            value=noisy(v, STD_V_PU),
            std_dev=STD_V_PU,
            element=int(bus),
            name=f"V_bus_{bus}",
        )

    # ---------- Bus P/Q injection measurements ----------
    for bus in net.bus.index:
        p = float(net.res_bus.at[bus, "p_mw"])
        q = float(net.res_bus.at[bus, "q_mvar"])

        pp.create_measurement(
            net,
            meas_type="p",
            element_type="bus",
            value=noisy(p, STD_P_MW),
            std_dev=STD_P_MW,
            element=int(bus),
            name=f"P_bus_{bus}",
        )

        pp.create_measurement(
            net,
            meas_type="q",
            element_type="bus",
            value=noisy(q, STD_Q_MVAR),
            std_dev=STD_Q_MVAR,
            element=int(bus),
            name=f"Q_bus_{bus}",
        )

    # ---------- Line P/Q measurements ----------
    for line in net.line.index:
        p_from = float(net.res_line.at[line, "p_from_mw"])
        q_from = float(net.res_line.at[line, "q_from_mvar"])

        pp.create_measurement(
            net,
            meas_type="p",
            element_type="line",
            value=noisy(p_from, STD_LINE_P_MW),
            std_dev=STD_LINE_P_MW,
            element=int(line),
            side="from",
            name=f"P_line_{line}_from",
        )

        pp.create_measurement(
            net,
            meas_type="q",
            element_type="line",
            value=noisy(q_from, STD_LINE_Q_MVAR),
            std_dev=STD_LINE_Q_MVAR,
            element=int(line),
            side="from",
            name=f"Q_line_{line}_from",
        )


def corrupt_by_name(net, name, delta):
    mask = net.measurement["name"] == name
    if not mask.any():
        raise KeyError(f"Measurement not found: {name}")

    idx = net.measurement.index[mask][0]
    net.measurement.at[idx, "value"] += delta
    return int(idx)


def apply_measurement_corruption(net, scenario_name):
    """
    Intentionally obvious/non-stealth corruption cases for defensive BDD testing.

    This phase does NOT construct an evasion-optimized attack.
    The goal is to establish a conventional AC state-estimation baseline.
    """
    corrupted = []

    if scenario_name == "single_voltage_corruption":
        # +8 % absolute voltage bias at remote Bus 4.
        corrupted.append(
            corrupt_by_name(net, "V_bus_4", +0.080)
        )

    elif scenario_name == "single_line_flow_corruption":
        # +25 kW bias on one line active-power measurement.
        corrupted.append(
            corrupt_by_name(net, "P_line_1_from", +0.025)
        )

    elif scenario_name == "coordinated_measurement_corruption":
        # Several related but deliberately non-stealth corruptions.
        corrupted.append(
            corrupt_by_name(net, "V_bus_3", -0.050)
        )
        corrupted.append(
            corrupt_by_name(net, "V_bus_4", -0.060)
        )
        corrupted.append(
            corrupt_by_name(net, "P_bus_4", +0.018)
        )
        corrupted.append(
            corrupt_by_name(net, "Q_bus_4", +0.010)
        )
        corrupted.append(
            corrupt_by_name(net, "P_line_3_from", +0.020)
        )

    return corrupted


def measurement_normalized_residuals(net):
    """
    Defensive post-estimation residual approximation for visualization.

    For bus V/P/Q and line P/Q, compare each measurement against the
    corresponding estimated quantity and normalize by its std_dev.

    This is not pandapower's internal largest-normalized-residual statistic;
    it is an interpretable measurement-vs-estimate diagnostic.
    """
    rows = []

    for idx, m in net.measurement.iterrows():
        mt = m["measurement_type"]
        et = m["element_type"]
        element = int(m["element"])
        value = float(m["value"])
        std = float(m["std_dev"])
        name = str(m["name"])

        estimate_value = np.nan

        if et == "bus":
            if mt == "v":
                estimate_value = float(net.res_bus_est.at[element, "vm_pu"])
            elif mt == "p":
                estimate_value = float(net.res_bus_est.at[element, "p_mw"])
            elif mt == "q":
                estimate_value = float(net.res_bus_est.at[element, "q_mvar"])

        elif et == "line":
            side = m["side"]

            if side not in ("from", "to"):
                # In this benchmark all line measurements use side="from".
                side = "from"

            if mt == "p":
                col = "p_from_mw" if side == "from" else "p_to_mw"
                estimate_value = float(net.res_line_est.at[element, col])

            elif mt == "q":
                col = "q_from_mvar" if side == "from" else "q_to_mvar"
                estimate_value = float(net.res_line_est.at[element, col])

        if np.isfinite(estimate_value):
            residual = value - estimate_value
            normalized = abs(residual) / max(std, 1e-12)

            rows.append({
                "measurement_index": int(idx),
                "name": name,
                "measurement_type": mt,
                "element_type": et,
                "element": element,
                "measured": value,
                "estimated": estimate_value,
                "std_dev": std,
                "residual": residual,
                "abs_normalized_residual": normalized,
            })

    return pd.DataFrame(rows)


def state_error_metrics(net):
    true_v = net.res_bus["vm_pu"].to_numpy(dtype=float)
    est_v = net.res_bus_est["vm_pu"].to_numpy(dtype=float)

    true_a = net.res_bus["va_degree"].to_numpy(dtype=float)
    est_a = net.res_bus_est["va_degree"].to_numpy(dtype=float)

    v_rmse = float(np.sqrt(np.mean((est_v - true_v)**2)))
    a_rmse = float(np.sqrt(np.mean((est_a - true_a)**2)))

    return v_rmse, a_rmse


def run_once(meta, repetition):
    net = build_truth_network(meta["name"])

    add_measurements_from_truth(net)
    corrupted_indices = apply_measurement_corruption(
        net,
        meta["name"],
    )

    n_measurements = len(net.measurement)

    try:
        success = bool(
            estimate(
                net,
                algorithm="wls",
                init="flat",
                tolerance=1e-7,
                maximum_iterations=30,
                calculate_voltage_angles=True,
            )
        )
    except Exception:
        success = False

    if success:
        v_rmse, angle_rmse = state_error_metrics(net)

        residuals = measurement_normalized_residuals(net)
        max_norm_res = (
            float(residuals["abs_normalized_residual"].max())
            if not residuals.empty else np.nan
        )
        worst_name = (
            str(
                residuals.loc[
                    residuals["abs_normalized_residual"].idxmax(),
                    "name",
                ]
            )
            if not residuals.empty else ""
        )

        # pandapower's built-in chi-squared bad-data test.
        try:
            chi2_flag = bool(
                chi2_analysis(
                    net,
                    init="flat",
                    tolerance=1e-7,
                    maximum_iterations=30,
                    calculate_voltage_angles=True,
                    chi2_prob_false=0.05,
                )
            )
        except Exception:
            chi2_flag = False
    else:
        v_rmse = np.nan
        angle_rmse = np.nan
        max_norm_res = np.nan
        worst_name = ""
        chi2_flag = False
        residuals = pd.DataFrame()

    cyber_truth = meta["family"] == "cyber"

    summary = {
        "scenario": meta["name"],
        "family": meta["family"],
        "repetition": repetition,
        "n_measurements": n_measurements,
        "n_corrupted_measurements": len(corrupted_indices),
        "estimation_success": success,
        "chi2_bad_data_flag": chi2_flag,
        "cyber_ground_truth": cyber_truth,
        "correct_chi2_classification": (
            chi2_flag if cyber_truth else not chi2_flag
        ),
        "voltage_rmse_pu": v_rmse,
        "angle_rmse_deg": angle_rmse,
        "max_abs_normalized_residual": max_norm_res,
        "worst_residual_measurement": worst_name,
    }

    if not residuals.empty:
        residuals["scenario"] = meta["name"]
        residuals["repetition"] = repetition
        residuals["is_intentionally_corrupted"] = residuals[
            "measurement_index"
        ].isin(corrupted_indices)

    return summary, residuals


def aggregate_results(summary_df):
    agg = summary_df.groupby(
        ["scenario", "family"],
        as_index=False,
    ).agg(
        runs=("repetition", "count"),
        estimator_success_rate=("estimation_success", "mean"),
        chi2_detection_rate=("chi2_bad_data_flag", "mean"),
        correct_classification_rate=("correct_chi2_classification", "mean"),
        mean_voltage_rmse_pu=("voltage_rmse_pu", "mean"),
        p95_voltage_rmse_pu=(
            "voltage_rmse_pu",
            lambda x: np.nanpercentile(x, 95)
        ),
        mean_angle_rmse_deg=("angle_rmse_deg", "mean"),
        mean_max_normalized_residual=(
            "max_abs_normalized_residual",
            "mean",
        ),
    )

    return agg


def pretty_name(s):
    return {
        "clean_noisy": "Normal + noise",
        "physical_load_step": "Physical load step",
        "single_voltage_corruption": "Voltage corruption",
        "single_line_flow_corruption": "Line-flow corruption",
        "coordinated_measurement_corruption": "Coordinated corruption",
    }[s]


def plot_detection_rate(agg):
    fig, ax = plt.subplots(figsize=(10.5, 6.0))

    x = np.arange(len(agg))
    vals = 100.0 * agg["chi2_detection_rate"].to_numpy()

    ax.bar(x, vals)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [pretty_name(x) for x in agg["scenario"]],
        rotation=20,
        ha="right",
    )

    for i, v in enumerate(vals):
        ax.text(i, v + 2, f"{v:.0f}%", ha="center", fontsize=10)

    ax.set_ylim(0, 110)
    ax.set_ylabel("χ² bad-data detection rate (%)")
    ax.set_title(
        "AC State-Estimation Baseline: Physical Events vs Measurement Corruption"
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(
        FIGURES / "09_ac_state_estimation_chi2_detection.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_state_error(agg):
    fig, ax = plt.subplots(figsize=(10.5, 6.0))

    x = np.arange(len(agg))
    vals = agg["mean_voltage_rmse_pu"].to_numpy()

    ax.bar(x, vals)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [pretty_name(x) for x in agg["scenario"]],
        rotation=20,
        ha="right",
    )

    ax.set_ylabel("Mean voltage-state RMSE (pu)")
    ax.set_title(
        "Impact of Corrupted Measurements on WLS Voltage-State Estimation"
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(
        FIGURES / "10_wls_voltage_state_error.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_residual_example(residual_df):
    target = residual_df[
        (residual_df["scenario"] == "coordinated_measurement_corruption")
        & (residual_df["repetition"] == 0)
    ].copy()

    if target.empty:
        return

    target = target.sort_values(
        "abs_normalized_residual",
        ascending=False,
    ).head(12)

    fig, ax = plt.subplots(figsize=(10.5, 6.4))

    y = np.arange(len(target))
    ax.barh(
        y,
        target["abs_normalized_residual"].to_numpy(),
    )
    ax.set_yticks(y)
    ax.set_yticklabels(target["name"].tolist())
    ax.invert_yaxis()

    ax.set_xlabel("|measurement − estimate| / σ")
    ax.set_title(
        "Largest Measurement-to-Estimate Residuals Under Coordinated Corruption"
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(
        FIGURES / "11_measurement_residual_ranking.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)


if __name__ == "__main__":
    # Monte-Carlo repetitions are essential: one noisy realization is not
    # enough to claim detector performance.
    N_REPEATS = 50

    summaries = []
    residual_frames = []

    print("\n=== PHASE 2C: AC STATE ESTIMATION + BDD BASELINE ===")
    print(f"Monte-Carlo repetitions per scenario: {N_REPEATS}")

    for meta in SCENARIOS:
        print(f"\nRunning {meta['name']} ...")

        for rep in range(N_REPEATS):
            summary, residuals = run_once(meta, rep)
            summaries.append(summary)

            if not residuals.empty and rep == 0:
                residual_frames.append(residuals)

        tmp = pd.DataFrame(
            [s for s in summaries if s["scenario"] == meta["name"]]
        )

        print(
            f"  estimator success = {100*tmp['estimation_success'].mean():.1f}%"
            f" | chi2 flags = {100*tmp['chi2_bad_data_flag'].mean():.1f}%"
            f" | correct = {100*tmp['correct_chi2_classification'].mean():.1f}%"
            f" | mean V RMSE = {tmp['voltage_rmse_pu'].mean():.5f} pu"
        )

    summary_df = pd.DataFrame(summaries)
    residual_df = (
        pd.concat(residual_frames, ignore_index=True)
        if residual_frames else pd.DataFrame()
    )

    agg = aggregate_results(summary_df)

    summary_df.to_csv(
        RESULTS / "phase2c_monte_carlo_runs.csv",
        index=False,
    )
    agg.to_csv(
        RESULTS / "phase2c_state_estimation_summary.csv",
        index=False,
    )

    if not residual_df.empty:
        residual_df.to_csv(
            RESULTS / "phase2c_example_residuals.csv",
            index=False,
        )

    plot_detection_rate(agg)
    plot_state_error(agg)

    if not residual_df.empty:
        plot_residual_example(residual_df)

    print("\n=== AGGREGATED RESULTS ===")
    print(
        agg[
            [
                "scenario",
                "family",
                "estimator_success_rate",
                "chi2_detection_rate",
                "correct_classification_rate",
                "mean_voltage_rmse_pu",
                "mean_angle_rmse_deg",
                "mean_max_normalized_residual",
            ]
        ].to_string(index=False)
    )

    print("\nSaved:")
    print("  results/phase2c_monte_carlo_runs.csv")
    print("  results/phase2c_state_estimation_summary.csv")
    print("  results/phase2c_example_residuals.csv")
    print("  figures/09_ac_state_estimation_chi2_detection.png")
    print("  figures/10_wls_voltage_state_error.png")
    print("  figures/11_measurement_residual_ranking.png")

    print(
        "\nINTERPRETATION:\n"
        "This phase establishes a conventional AC WLS + chi-squared "
        "bad-data baseline under realistic measurement noise and redundancy. "
        "The next research step should stress this baseline with harder "
        "coherent/unseen corruption and compare it against robust/physics- "
        "and topology-aware defensive detectors, rather than relying on "
        "one deterministic threshold result."
    )
