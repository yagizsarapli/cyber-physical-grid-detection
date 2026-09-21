from pathlib import Path
import importlib.util
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
inv = load_module("02_inverter_models.py", "inverter_models")


# ============================================================
# PHASE 2B — CLOSED-LOOP CYBER-PHYSICAL BENCHMARK
# ============================================================
CONTROL_DT = 1e-3          # inverter-control integration step
NETWORK_DT = 20e-3         # quasi-static AC power-flow update
SIM_END = 4.0

EVENT_START = 1.0
EVENT_END = 1.25

S_BASE_MVA = 0.10

PV_P_REF_PU = 0.35
BESS_P_REF_AUTH_PU = 0.15

GFL_IMAX_AUTH_PU = 1.20

PHYSICAL_SAG_PU = 0.60
PHYSICAL_ANGLE_JUMP_DEG = 8.0

SENSOR_THRESHOLD = 1.0
COMMAND_THRESHOLD = 1.0
TOTAL_THRESHOLD = 1.0


SCENARIOS = [
    {
        "name": "normal",
        "family": "normal",
        "cyber": False,
        "description": "Normal grid-connected operation",
    },
    {
        "name": "physical_load_step",
        "family": "physical",
        "cyber": False,
        "description": "Real load increase at Bus 2",
    },
    {
        "name": "physical_voltage_sag",
        "family": "physical",
        "cyber": False,
        "description": "Real upstream voltage sag and angle jump",
    },
    {
        "name": "cyber_fake_sag",
        "family": "cyber",
        "cyber": True,
        "description": "Healthy grid, spoofed low-voltage/angle measurement at GFL",
    },
    {
        "name": "cyber_phase_drift",
        "family": "cyber",
        "cyber": True,
        "description": "Healthy grid, gradual phase/frequency-like sensor drift at GFL",
    },
    {
        "name": "cyber_gfl_current_limit",
        "family": "cyber",
        "cyber": True,
        "description": "Unauthorized reduction of GFL current-limit parameter",
    },
    {
        "name": "cyber_gfm_setpoint",
        "family": "cyber",
        "cyber": True,
        "description": "Unauthorized GFM active-power setpoint manipulation",
    },
    {
        "name": "hybrid_fault_masking",
        "family": "hybrid",
        "cyber": True,
        "description": "Real voltage sag hidden from GFL by falsified healthy measurement",
    },
]


def in_event(t):
    return EVENT_START <= t < EVENT_END


def wrap_angle(rad):
    return (rad + np.pi) % (2*np.pi) - np.pi


def configure_physical_system(net, scenario_name, t):
    """Apply physical-side events before each network solve."""
    ext = net.ext_grid.index[0]

    # Always restore nominal source first.
    net.ext_grid.at[ext, "vm_pu"] = 1.0
    net.ext_grid.at[ext, "va_degree"] = 0.0

    # Restore nominal loads.
    net.load.loc[:, "p_mw"] = [0.030, 0.025]
    net.load.loc[:, "q_mvar"] = [0.008, 0.006]

    if not in_event(t):
        return

    if scenario_name in ("physical_voltage_sag", "hybrid_fault_masking"):
        net.ext_grid.at[ext, "vm_pu"] = PHYSICAL_SAG_PU
        net.ext_grid.at[ext, "va_degree"] = PHYSICAL_ANGLE_JUMP_DEG

    elif scenario_name == "physical_load_step":
        # Bus-2 load increases from 30 kW to 55 kW.
        idx = net.load.index[0]
        net.load.at[idx, "p_mw"] = 0.055
        net.load.at[idx, "q_mvar"] = 0.014


def attacked_measurement(
    scenario_name,
    t,
    v_true_pu,
    theta_true_rad,
):
    """Return the voltage/angle channel seen by the PV-GFL controller."""
    v_meas = v_true_pu
    theta_meas = theta_true_rad

    if not in_event(t):
        return v_meas, theta_meas

    if scenario_name == "cyber_fake_sag":
        v_meas = 0.60
        theta_meas = np.deg2rad(8.0)

    elif scenario_name == "cyber_phase_drift":
        # 0.5-Hz-equivalent angle bias accumulated during attack.
        dt_attack = t - EVENT_START
        theta_meas = theta_true_rad + 2*np.pi*0.5*dt_attack

    elif scenario_name == "hybrid_fault_masking":
        # A real physical sag exists, but the compromised sensor reports healthy grid.
        v_meas = 1.0
        theta_meas = 0.0

    return v_meas, theta_meas


def cyber_control_parameters(scenario_name, t):
    """Return parameters actually used by controllers and their authorized values."""
    gfl_imax_used = GFL_IMAX_AUTH_PU
    gfm_pref_used = BESS_P_REF_AUTH_PU

    if in_event(t):
        if scenario_name == "cyber_gfl_current_limit":
            # Deliberately below normal PV current demand (~0.35 pu).
            gfl_imax_used = 0.25

        elif scenario_name == "cyber_gfm_setpoint":
            # Unauthorized dispatch change.
            gfm_pref_used = 0.65

    return gfl_imax_used, gfm_pref_used


def run_scenario(meta):
    scenario = meta["name"]

    net = topology.build_microgrid()
    pv_bus = net.gridra["bus_pv"]
    bess_bus = net.gridra["bus_bess"]
    pv_idx = net.gridra["pv_sgen"]
    bess_idx = net.gridra["bess_sgen"]

    gfl = inv.GFLInverter(
        kp_pll=18.0,
        ki_pll=250.0,
        i_max_pu=GFL_IMAX_AUTH_PU,
        q_voltage_gain=2.0,
    )
    gfm = inv.GFMInverter(
        m_inertia=0.25,
        d_damping=0.80,
        e_internal_pu=1.02,
        x_virtual_pu=0.25,
        i_max_pu=1.20,
    )

    # Initial AC solution using nominal injections.
    topology.solve_base_case(net)

    theta_pv_0 = np.deg2rad(float(net.res_bus.at[pv_bus, "va_degree"]))
    theta_bess_0 = np.deg2rad(float(net.res_bus.at[bess_bus, "va_degree"]))

    gfl.reset(theta0=theta_pv_0)

    # Use a power-angle equilibrium approximation around the solved BESS-bus angle.
    arg = BESS_P_REF_AUTH_PU * gfm.x_virtual_pu / gfm.e_internal_pu
    delta0 = np.arcsin(np.clip(arg, -0.95, 0.95))
    gfm.reset(theta0=theta_bess_0 + delta0)

    # Last network-visible converter injections.
    pv_p_mw = 0.035
    pv_q_mvar = 0.0
    bess_p_mw = 0.015
    bess_q_mvar = 0.0

    # Cached physical states between quasi-static network solves.
    v_pv_true = float(net.res_bus.at[pv_bus, "vm_pu"])
    theta_pv_true = theta_pv_0
    v_bess_true = float(net.res_bus.at[bess_bus, "vm_pu"])
    theta_bess_true = theta_bess_0

    network_stride = max(1, int(round(NETWORK_DT / CONTROL_DT)))

    rows = []

    for k, t in enumerate(np.arange(0.0, SIM_END + CONTROL_DT/2, CONTROL_DT)):
        # ------------------------------------------------------------
        # 1) PHYSICAL NETWORK UPDATE
        # ------------------------------------------------------------
        if k % network_stride == 0:
            configure_physical_system(net, scenario, t)

            net.sgen.at[pv_idx, "p_mw"] = pv_p_mw
            net.sgen.at[pv_idx, "q_mvar"] = pv_q_mvar
            net.sgen.at[bess_idx, "p_mw"] = bess_p_mw
            net.sgen.at[bess_idx, "q_mvar"] = bess_q_mvar

            try:
                topology.solve_base_case(net)
            except Exception as exc:
                raise RuntimeError(
                    f"Power flow failed in scenario={scenario}, t={t:.3f}s"
                ) from exc

            v_pv_true = float(net.res_bus.at[pv_bus, "vm_pu"])
            theta_pv_true = np.deg2rad(
                float(net.res_bus.at[pv_bus, "va_degree"])
            )

            v_bess_true = float(net.res_bus.at[bess_bus, "vm_pu"])
            theta_bess_true = np.deg2rad(
                float(net.res_bus.at[bess_bus, "va_degree"])
            )

            ext_p_mw = float(net.res_ext_grid.iloc[0]["p_mw"])
            ext_q_mvar = float(net.res_ext_grid.iloc[0]["q_mvar"])
            min_bus_v = float(net.res_bus["vm_pu"].min())
            max_line_loading = float(net.res_line["loading_percent"].max())

        # ------------------------------------------------------------
        # 2) SENSOR / COMMAND CYBER LAYER
        # ------------------------------------------------------------
        v_gfl_meas, theta_gfl_meas = attacked_measurement(
            scenario,
            t,
            v_pv_true,
            theta_pv_true,
        )

        gfl_imax_used, gfm_pref_used = cyber_control_parameters(
            scenario,
            t,
        )

        gfl.i_max_pu = gfl_imax_used

        # ------------------------------------------------------------
        # 3) INVERTER INTERNAL DYNAMICS
        # ------------------------------------------------------------
        gfl_out = gfl.step(
            v_meas_pu=v_gfl_meas,
            theta_meas_rad=theta_gfl_meas,
            p_ref_pu=PV_P_REF_PU,
            q_ref_pu=0.0,
            dt=CONTROL_DT,
        )

        gfm_out = gfm.step(
            v_pcc_pu=v_bess_true,
            theta_pcc_rad=theta_bess_true,
            p_ref_pu=gfm_pref_used,
            dt=CONTROL_DT,
        )

        # ------------------------------------------------------------
        # 4) CLOSE THE LOOP: controller outputs -> network injections
        # ------------------------------------------------------------
        pv_p_mw = gfl_out["p_out_pu"] * S_BASE_MVA

        # GFL model uses an internal iq sign convention where positive iq is
        # voltage-support current; convert it to positive reactive injection.
        pv_q_mvar = -gfl_out["q_out_pu"] * S_BASE_MVA

        bess_p_mw = gfm_out["p_out_pu"] * S_BASE_MVA
        bess_q_mvar = gfm_out["q_out_pu"] * S_BASE_MVA

        # ------------------------------------------------------------
        # 5) MODEL-/INTEGRITY-BASED RESIDUALS
        # ------------------------------------------------------------
        # Digital-twin sensor residual:
        # compare compromised GFL measurement with the AC-network state
        # predicted by the closed-loop model. This removes the explicit
        # "second trusted voltage sensor" used in Phase 2A.
        dv = abs(v_gfl_meas - v_pv_true) / 0.03
        dtheta_deg = abs(
            np.rad2deg(wrap_angle(theta_gfl_meas - theta_pv_true))
        )
        dtheta = dtheta_deg / 3.0
        sensor_residual = dv + dtheta

        # Controller-command integrity residual:
        # compare active settings with their authorized baseline.
        imax_residual = abs(gfl_imax_used - GFL_IMAX_AUTH_PU) / 0.10
        pref_residual = abs(gfm_pref_used - BESS_P_REF_AUTH_PU) / 0.05
        command_residual = imax_residual + pref_residual

        # Hybrid detection score.
        total_score = max(sensor_residual, command_residual)
        detected = total_score >= TOTAL_THRESHOLD

        rows.append({
            "scenario": scenario,
            "family": meta["family"],
            "cyber_ground_truth": bool(meta["cyber"]),
            "description": meta["description"],
            "time_s": t,
            "event_active": bool(in_event(t)),

            "v_pv_true_pu": v_pv_true,
            "theta_pv_true_deg": np.rad2deg(theta_pv_true),
            "v_gfl_measured_pu": v_gfl_meas,
            "theta_gfl_measured_deg": np.rad2deg(theta_gfl_meas),

            "v_bess_true_pu": v_bess_true,
            "theta_bess_true_deg": np.rad2deg(theta_bess_true),

            "min_bus_v_pu": min_bus_v,
            "max_line_loading_percent": max_line_loading,
            "ext_grid_p_mw": ext_p_mw,
            "ext_grid_q_mvar": ext_q_mvar,

            "gfl_pll_angle_deg": gfl_out["theta_pll_deg"],
            "gfl_pll_freq_dev_hz": gfl_out["pll_freq_dev_hz"],
            "gfl_id_pu": gfl_out["id_pu"],
            "gfl_iq_pu": gfl_out["iq_pu"],
            "gfl_current_pu": gfl_out["i_mag_pu"],
            "gfl_limiter": bool(gfl_out["limiter_active"]),
            "gfl_imax_used_pu": gfl_imax_used,

            "gfm_angle_deg": gfm_out["theta_internal_deg"],
            "gfm_freq_dev_hz": gfm_out["gfm_freq_dev_hz"],
            "gfm_p_pu": gfm_out["p_out_pu"],
            "gfm_q_pu": gfm_out["q_out_pu"],
            "gfm_current_pu": gfm_out["i_mag_pu"],
            "gfm_limiter": bool(gfm_out["limiter_active"]),
            "gfm_pref_used_pu": gfm_pref_used,

            "sensor_residual": sensor_residual,
            "command_residual": command_residual,
            "detection_score": total_score,
            "detected": bool(detected),
        })

    return pd.DataFrame(rows)


def first_detection_delay(df):
    event_df = df[df["event_active"]]
    hit = event_df[event_df["detected"]]
    if hit.empty:
        return np.nan
    return float(hit.iloc[0]["time_s"] - EVENT_START)


def summarize_scenario(df):
    event = df[df["event_active"]]
    cyber_truth = bool(df.iloc[0]["cyber_ground_truth"])
    detected_any = bool(event["detected"].any())

    return {
        "scenario": df.iloc[0]["scenario"],
        "family": df.iloc[0]["family"],
        "cyber_ground_truth": cyber_truth,
        "detected_during_event": detected_any,
        "correct_detection": (
            detected_any if cyber_truth else not detected_any
        ),
        "detection_delay_ms": (
            first_detection_delay(df) * 1000.0
            if cyber_truth else np.nan
        ),
        "max_detection_score": float(event["detection_score"].max()),
        "min_bus_voltage_pu": float(event["min_bus_v_pu"].min()),
        "max_line_loading_percent": float(
            event["max_line_loading_percent"].max()
        ),
        "max_abs_gfl_pll_freq_hz": float(
            event["gfl_pll_freq_dev_hz"].abs().max()
        ),
        "max_gfl_current_pu": float(event["gfl_current_pu"].max()),
        "gfl_limiter_seen": bool(event["gfl_limiter"].any()),
        "max_abs_gfm_freq_hz": float(
            event["gfm_freq_dev_hz"].abs().max()
        ),
        "max_gfm_current_pu": float(event["gfm_current_pu"].max()),
        "gfm_limiter_seen": bool(event["gfm_limiter"].any()),
    }


def pretty_name(name):
    return {
        "normal": "Normal",
        "physical_load_step": "Physical load step",
        "physical_voltage_sag": "Physical voltage sag",
        "cyber_fake_sag": "Fake voltage sag",
        "cyber_phase_drift": "Phase-drift attack",
        "cyber_gfl_current_limit": "GFL current-limit attack",
        "cyber_gfm_setpoint": "GFM setpoint attack",
        "hybrid_fault_masking": "Fault-masking attack",
    }[name]


def figure_detection_scores(summary):
    fig, ax = plt.subplots(figsize=(12, 6.2))

    names = [pretty_name(x) for x in summary["scenario"]]
    vals = summary["max_detection_score"].to_numpy()

    x = np.arange(len(names))
    ax.bar(x, vals)

    ax.axhline(TOTAL_THRESHOLD, linestyle="--", linewidth=1.3)
    ax.text(
        len(names)-0.55,
        TOTAL_THRESHOLD + 0.18,
        "Detection threshold",
        ha="right",
        fontsize=10,
    )

    for i, v in enumerate(vals):
        ax.text(i, v + 0.12, f"{v:.1f}", ha="center", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=24, ha="right")
    ax.set_ylabel("Maximum detection score during event")
    ax.set_title(
        "Closed-Loop Cyber-Physical Detection Across Physical, Cyber and Hybrid Events"
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(
        FIGURES / "05_closed_loop_detection_benchmark.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)


def figure_response_fingerprint(summary):
    fig, ax = plt.subplots(figsize=(9, 6.6))

    x = summary["max_abs_gfl_pll_freq_hz"].to_numpy()
    y = summary["max_abs_gfm_freq_hz"].to_numpy()

    ax.scatter(x, y, s=85)

    for i, row in summary.iterrows():
        ax.annotate(
            pretty_name(row["scenario"]),
            (row["max_abs_gfl_pll_freq_hz"], row["max_abs_gfm_freq_hz"]),
            xytext=(6, 5),
            textcoords="offset points",
            fontsize=9,
        )

    ax.set_xlabel("Maximum |GFL PLL frequency deviation| (Hz)")
    ax.set_ylabel("Maximum |GFM frequency deviation| (Hz)")
    ax.set_title("Inverter Response Fingerprints of Physical and Cyber Events")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(
        FIGURES / "06_inverter_response_fingerprint.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)


def figure_fault_masking(all_data):
    d = all_data[all_data["scenario"] == "hybrid_fault_masking"]

    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.plot(
        d["time_s"], d["v_pv_true_pu"],
        linewidth=2.3, label="Physical PV-bus voltage"
    )
    ax.plot(
        d["time_s"], d["v_gfl_measured_pu"],
        linewidth=2.3, linestyle="--",
        label="Compromised GFL measurement"
    )
    ax.axvspan(EVENT_START, EVENT_END, alpha=0.10)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Voltage (pu)")
    ax.set_title(
        "Hybrid Fault-Masking Attack: A Real Grid Disturbance Hidden from the GFL Controller"
    )
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(
        FIGURES / "07_hybrid_fault_masking_voltage.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)


def figure_fault_masking_iq(all_data):
    d = all_data[all_data["scenario"] == "hybrid_fault_masking"]

    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.plot(
        d["time_s"], d["gfl_iq_pu"],
        linewidth=2.3,
        label="GFL reactive-current command"
    )
    ax.axvspan(EVENT_START, EVENT_END, alpha=0.10)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("GFL iq (pu)")
    ax.set_title(
        "Control Consequence of Fault Masking: Suppressed GFL Voltage-Support Response"
    )
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(
        FIGURES / "08_fault_masking_gfl_reactive_response.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)


if __name__ == "__main__":
    all_frames = []
    summaries = []

    print("\n=== PHASE 2B: CLOSED-LOOP CYBER-PHYSICAL BENCHMARK ===")

    for meta in SCENARIOS:
        print(f"\nRunning {meta['name']} ...")
        df = run_scenario(meta)
        all_frames.append(df)

        s = summarize_scenario(df)
        summaries.append(s)

        print(
            f"  detected={s['detected_during_event']} | "
            f"correct={s['correct_detection']} | "
            f"score={s['max_detection_score']:.2f} | "
            f"Vmin={s['min_bus_voltage_pu']:.3f} pu | "
            f"|df_PLL|max={s['max_abs_gfl_pll_freq_hz']:.3f} Hz | "
            f"|df_GFM|max={s['max_abs_gfm_freq_hz']:.3f} Hz"
        )

    all_data = pd.concat(all_frames, ignore_index=True)
    summary = pd.DataFrame(summaries)

    all_data.to_csv(
        RESULTS / "phase2b_closed_loop_timeseries.csv",
        index=False,
    )
    summary.to_csv(
        RESULTS / "phase2b_scenario_summary.csv",
        index=False,
    )

    figure_detection_scores(summary)
    figure_response_fingerprint(summary)
    figure_fault_masking(all_data)
    figure_fault_masking_iq(all_data)

    print("\n=== SUMMARY ===")
    cols = [
        "scenario",
        "family",
        "cyber_ground_truth",
        "detected_during_event",
        "correct_detection",
        "detection_delay_ms",
        "max_detection_score",
        "min_bus_voltage_pu",
        "max_abs_gfl_pll_freq_hz",
        "max_abs_gfm_freq_hz",
    ]
    print(summary[cols].to_string(index=False))

    accuracy = summary["correct_detection"].mean()
    print(f"\nScenario-level classification accuracy: {100*accuracy:.1f}%")

    print("\nSaved:")
    print("  results/phase2b_closed_loop_timeseries.csv")
    print("  results/phase2b_scenario_summary.csv")
    print("  figures/05_closed_loop_detection_benchmark.png")
    print("  figures/06_inverter_response_fingerprint.png")
    print("  figures/07_hybrid_fault_masking_voltage.png")
    print("  figures/08_fault_masking_gfl_reactive_response.png")

    print(
        "\nNEXT LIMITATION TO REMOVE:\n"
        "This benchmark uses a closed-loop AC digital-twin residual and "
        "authorized controller settings. A sufficiently informed attacker "
        "could manipulate multiple measurements coherently and evade these "
        "simple residuals. Phase 2C will therefore implement AC state "
        "estimation / BDD and construct stealth-constrained FDIA cases "
        "specifically designed to bypass conventional residual detection."
    )
