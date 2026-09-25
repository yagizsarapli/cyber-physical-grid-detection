from pathlib import Path
import importlib.util
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
DATA.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)


def load_module(filename, module_name):
    spec = importlib.util.spec_from_file_location(module_name, ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


topology = load_module("01_microgrid_topology.py", "topology")


# ============================================================
# PHASE 2E — 7-DAY MULTI-RATE OPERATING DATASET BACKBONE
# ============================================================
# Slow layer:
#   7 days @ 5-minute resolution = 2016 operating points
#
# Later fast layer:
#   selected event timestamps will receive high-resolution
#   inverter-dynamic windows (GFL/GFM, PLL, current limiting, etc.).
#
# This avoids the physically wrong idea of integrating inverter dynamics
# with a 5-minute control timestep.
# ============================================================

RNG = np.random.default_rng(20260811)

N_DAYS = 7
STEP_MIN = 5
SAMPLES_PER_DAY = 24 * 60 // STEP_MIN
N_STEPS = N_DAYS * SAMPLES_PER_DAY
DT_H = STEP_MIN / 60.0

PV_RATED_MW = 0.040       # 40 kW
LOAD_A_BASE_MW = 0.030
LOAD_B_BASE_MW = 0.025

BESS_E_MWH = 0.080        # 80 kWh
BESS_P_MAX_MW = 0.020     # 20 kW
BESS_SOC_INIT = 0.55
BESS_SOC_MIN = 0.20
BESS_SOC_MAX = 0.85
ETA_CHARGE = 0.95
ETA_DISCHARGE = 0.95

# Simple dispatch target: reduce grid exchange around zero when possible.
GRID_TARGET_MW = 0.0

START_TIMESTAMP = "2026-08-03 00:00:00"


def smooth_cloud_factor(n):
    """
    Slowly varying normal weather variability only.

    IMPORTANT:
    Explicit physical cloud-transient events are NOT injected here.
    They are applied later in the high-resolution transient layer so
    matched normal/event counterfactuals start from the same baseline.
    """
    raw = RNG.normal(0.0, 0.045, n)

    # Smooth over ~45 minutes to create realistic slow irradiance drift
    # without inserting the labeled fast transient events themselves.
    kernel = np.ones(9) / 9.0
    smooth = np.convolve(raw, kernel, mode="same")

    # Add a gentle multi-day modulation.
    t = np.arange(n, dtype=float)
    multiday = 0.035 * np.sin(
        2.0 * np.pi * t / (2.8 * SAMPLES_PER_DAY)
    )

    factor = 0.96 + smooth + multiday
    return np.clip(factor, 0.78, 1.05)

def solar_profile(index):
    hod = (
        index.hour.to_numpy(dtype=float)
        + index.minute.to_numpy(dtype=float) / 60.0
    )

    # Approximate daylight 05:30–20:30, peak near 13:00.
    sunrise = 5.5
    sunset = 20.5

    solar = np.zeros(len(index), dtype=float)
    mask = (hod >= sunrise) & (hod <= sunset)

    phase = (hod[mask] - sunrise) / (sunset - sunrise)
    solar[mask] = np.sin(np.pi * phase) ** 1.45

    return solar


def load_multiplier(index, household_shift=0.0):
    hod = (
        index.hour.to_numpy(dtype=float)
        + index.minute.to_numpy(dtype=float) / 60.0
    )

    morning = 0.32 * np.exp(
        -0.5 * ((hod - (7.3 + household_shift)) / 1.7) ** 2
    )
    evening = 0.48 * np.exp(
        -0.5 * ((hod - (19.2 + household_shift)) / 2.2) ** 2
    )
    night = 0.08 * np.exp(
        -0.5 * ((hod - 1.0) / 2.5) ** 2
    )

    weekend = np.where(
        index.dayofweek.to_numpy(dtype=int) >= 5,
        1.08,
        1.0
    )

    profile = (
        0.70
        + morning
        + evening
        + night
    ) * weekend

    noise = RNG.normal(0.0, 0.025, len(index))
    return np.clip(profile + noise, 0.45, 1.55)


def reactive_from_pf(p_mw, pf=0.95):
    phi = np.arccos(pf)
    return p_mw * np.tan(phi)


def bess_dispatch(net_demand_without_bess_mw, soc, timestamp):
    """
    Reserve-aware rule-based EMS.

    Positive result = BESS injects power into AC grid (discharge).
    Negative result = BESS absorbs power (charge).

    Design goals
    ------------
    - preserve SOC diversity across the week
    - charge from genuine PV surplus when available
    - modestly shave evening peaks
    - allow limited off-peak grid charging if reserve becomes low
    - avoid driving the battery to its lower SOC limit all day

    This is an operating-context EMS, not an optimized market controller.
    """
    hod = timestamp.hour + timestamp.minute / 60.0

    p_cmd = 0.0

    # 1) Prefer charging from PV surplus.
    if net_demand_without_bess_mw < -0.002 and soc < BESS_SOC_MAX:
        surplus = -net_demand_without_bess_mw
        p_cmd = -min(surplus, 0.015, BESS_P_MAX_MW)

    # 2) Peak shaving only during the evening peak.
    elif (
        17.0 <= hod < 22.0
        and net_demand_without_bess_mw > 0.045
        and soc > 0.35
    ):
        shave = net_demand_without_bess_mw - 0.045
        p_cmd = min(shave, 0.015, BESS_P_MAX_MW)

    # 3) Maintain emergency/control reserve using slow off-peak charging.
    elif (
        (hod >= 0.0 and hod < 5.0)
        and soc < 0.45
    ):
        p_cmd = -min(0.006, BESS_P_MAX_MW)

    # Enforce energy limits over the next slow timestep.
    if p_cmd > 0.0:
        available_mwh = max(
            0.0,
            (soc - BESS_SOC_MIN) * BESS_E_MWH
        )
        max_from_soc = (
            available_mwh
            * ETA_DISCHARGE
            / DT_H
        )
        p_cmd = min(p_cmd, max_from_soc)

    elif p_cmd < 0.0:
        room_mwh = max(
            0.0,
            (BESS_SOC_MAX - soc) * BESS_E_MWH
        )
        max_charge_from_soc = (
            room_mwh
            / (ETA_CHARGE * DT_H)
        )
        p_cmd = max(p_cmd, -max_charge_from_soc)

    return float(p_cmd)

def update_soc(soc, p_bess_mw):
    if p_bess_mw >= 0.0:
        # Discharging: stored energy decreases.
        delta_mwh = -(p_bess_mw / ETA_DISCHARGE) * DT_H
    else:
        # Charging: stored energy increases.
        delta_mwh = (-p_bess_mw * ETA_CHARGE) * DT_H

    soc_new = soc + delta_mwh / BESS_E_MWH
    return float(np.clip(soc_new, BESS_SOC_MIN, BESS_SOC_MAX))


def schedule_event_labels(index):
    """
    Labels only. Fast cyber/physical transient windows will be generated
    in the next phase around these timestamps.
    """
    labels = np.array(["normal"] * len(index), dtype=object)

    event_specs = [
        # timestamp offset in slow samples, duration in samples, label
        (1*SAMPLES_PER_DAY + 11*12, 8, "physical_cloud_transient"),
        (2*SAMPLES_PER_DAY + 18*12, 3, "physical_load_spike"),
        (3*SAMPLES_PER_DAY + 15*12, 2, "physical_line_event"),
        (4*SAMPLES_PER_DAY + 12*12, 3, "cyber_fake_voltage_measurement"),
        (4*SAMPLES_PER_DAY + 17*12, 4, "cyber_phase_drift"),
        (5*SAMPLES_PER_DAY + 14*12, 3, "cyber_gfm_setpoint_manipulation"),
        (6*SAMPLES_PER_DAY + 10*12, 3, "hybrid_fault_masking"),
    ]

    for start, width, label in event_specs:
        stop = min(len(labels), start + width)
        if start < len(labels):
            labels[start:stop] = label

    return labels


def run():
    index = pd.date_range(
        START_TIMESTAMP,
        periods=N_STEPS,
        freq=f"{STEP_MIN}min",
    )

    solar = solar_profile(index)
    weather = smooth_cloud_factor(N_STEPS)

    pv_profile_mw = PV_RATED_MW * solar * weather

    load_a_mult = load_multiplier(index, household_shift=0.0)
    load_b_mult = load_multiplier(index, household_shift=0.6)

    load_a_p = np.asarray(
        LOAD_A_BASE_MW * load_a_mult,
        dtype=float
    ).copy()
    load_b_p = np.asarray(
        LOAD_B_BASE_MW * load_b_mult,
        dtype=float
    ).copy()

    labels = schedule_event_labels(index)

    net = topology.build_microgrid()

    pv_idx = net.gridra["pv_sgen"]
    bess_idx = net.gridra["bess_sgen"]

    load_a_idx = net.load.index[0]
    load_b_idx = net.load.index[1]

    soc = BESS_SOC_INIT
    rows = []

    previous_pf_ok = True

    for k, ts in enumerate(index):
        p_pv = float(pv_profile_mw[k])

        p_la = float(load_a_p[k])
        p_lb = float(load_b_p[k])

        q_la = float(reactive_from_pf(p_la, pf=0.95))
        q_lb = float(reactive_from_pf(p_lb, pf=0.95))

        net_without_bess = p_la + p_lb - p_pv
        p_bess = bess_dispatch(net_without_bess, soc, ts)

        # In this slow operating layer BESS is unity-PF.
        q_bess = 0.0

        net.load.at[load_a_idx, "p_mw"] = p_la
        net.load.at[load_a_idx, "q_mvar"] = q_la

        net.load.at[load_b_idx, "p_mw"] = p_lb
        net.load.at[load_b_idx, "q_mvar"] = q_lb

        net.sgen.at[pv_idx, "p_mw"] = p_pv
        net.sgen.at[pv_idx, "q_mvar"] = 0.0

        net.sgen.at[bess_idx, "p_mw"] = p_bess
        net.sgen.at[bess_idx, "q_mvar"] = q_bess

        try:
            topology.solve_base_case(net)
            pf_ok = True
        except Exception:
            pf_ok = False

        if not pf_ok:
            rows.append({
                "timestamp": ts,
                "step": k,
                "event_label": labels[k],
                "powerflow_ok": False,
            })
            previous_pf_ok = False
            continue

        # Save bus states.
        row = {
            "timestamp": ts,
            "step": k,
            "event_label": labels[k],
            "powerflow_ok": True,

            "pv_p_mw": p_pv,
            "load_a_p_mw": p_la,
            "load_b_p_mw": p_lb,
            "load_total_p_mw": p_la + p_lb,

            "bess_p_mw": p_bess,
            "bess_soc": soc,

            "ext_grid_p_mw": float(net.res_ext_grid.iloc[0]["p_mw"]),
            "ext_grid_q_mvar": float(net.res_ext_grid.iloc[0]["q_mvar"]),

            "min_bus_v_pu": float(net.res_bus["vm_pu"].min()),
            "max_bus_v_pu": float(net.res_bus["vm_pu"].max()),
            "max_line_loading_percent": float(
                net.res_line["loading_percent"].max()
            ),
        }

        for bus in net.bus.index:
            row[f"bus_{bus}_vm_pu"] = float(
                net.res_bus.at[bus, "vm_pu"]
            )
            row[f"bus_{bus}_va_deg"] = float(
                net.res_bus.at[bus, "va_degree"]
            )

        for line in net.line.index:
            row[f"line_{line}_loading_pct"] = float(
                net.res_line.at[line, "loading_percent"]
            )
            row[f"line_{line}_p_from_mw"] = float(
                net.res_line.at[line, "p_from_mw"]
            )
            row[f"line_{line}_q_from_mvar"] = float(
                net.res_line.at[line, "q_from_mvar"]
            )

        rows.append(row)

        # SOC advances after the current interval.
        soc = update_soc(soc, p_bess)
        previous_pf_ok = True

    df = pd.DataFrame(rows)

    # Event calendar for next high-resolution phase.
    event_calendar = (
        df[df["event_label"] != "normal"][
            ["timestamp", "step", "event_label"]
        ]
        .copy()
        .reset_index(drop=True)
    )

    # Add day/hour columns useful for ML later.
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["day_index"] = (
        (df["timestamp"] - df["timestamp"].min())
        .dt.total_seconds()
        / 86400.0
    )
    df["hour_of_day"] = (
        df["timestamp"].dt.hour
        + df["timestamp"].dt.minute / 60.0
    )

    df.to_csv(
        DATA / "phase2e_7day_operating_dataset.csv",
        index=False,
    )

    event_calendar.to_csv(
        DATA / "phase2e_fast_event_calendar.csv",
        index=False,
    )

    # Summary.
    summary = pd.DataFrame([
        {
            "n_steps": len(df),
            "resolution_min": STEP_MIN,
            "powerflow_success_rate": float(
                df["powerflow_ok"].mean()
            ),
            "pv_energy_mwh": float(
                df["pv_p_mw"].sum() * DT_H
            ),
            "load_energy_mwh": float(
                df["load_total_p_mw"].sum() * DT_H
            ),
            "grid_import_energy_mwh": float(
                np.clip(df["ext_grid_p_mw"], 0, None).sum() * DT_H
            ),
            "grid_export_energy_mwh": float(
                -np.clip(df["ext_grid_p_mw"], None, 0).sum() * DT_H
            ),
            "min_bus_voltage_pu": float(
                df["min_bus_v_pu"].min()
            ),
            "max_line_loading_percent": float(
                df["max_line_loading_percent"].max()
            ),
            "min_soc": float(df["bess_soc"].min()),
            "max_soc": float(df["bess_soc"].max()),
            "mean_soc": float(df["bess_soc"].mean()),
            "soc_std": float(df["bess_soc"].std()),
        }
    ])

    summary.to_csv(
        RESULTS / "phase2e_operating_summary.csv",
        index=False,
    )

    # Figure 1 — 7-day power operation.
    fig, ax = plt.subplots(figsize=(13, 6.3))
    x_days = df["day_index"]

    ax.plot(
        x_days,
        1000*df["pv_p_mw"],
        linewidth=1.8,
        label="PV generation"
    )
    ax.plot(
        x_days,
        1000*df["load_total_p_mw"],
        linewidth=1.8,
        label="Total load"
    )
    ax.plot(
        x_days,
        1000*df["bess_p_mw"],
        linewidth=1.6,
        label="BESS power (+ discharge)"
    )
    ax.plot(
        x_days,
        1000*df["ext_grid_p_mw"],
        linewidth=1.5,
        label="Grid exchange (+ import)"
    )

    ax.axhline(0.0, linewidth=0.8)
    ax.set_xlabel("Simulation day")
    ax.set_ylabel("Active power (kW)")
    ax.set_title(
        "Seven-Day Microgrid Operation — PV, Load, BESS and Grid Exchange"
    )
    ax.legend(frameon=False, ncol=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(
        FIGURES / "16_7day_microgrid_operation.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)

    # Figure 2 — SOC.
    fig, ax = plt.subplots(figsize=(13, 5.3))
    ax.plot(
        x_days,
        100*df["bess_soc"],
        linewidth=2.0,
    )
    ax.axhline(
        100*BESS_SOC_MIN,
        linestyle="--",
        linewidth=1.0,
        label="SOC limits"
    )
    ax.axhline(
        100*BESS_SOC_MAX,
        linestyle="--",
        linewidth=1.0,
    )
    ax.set_xlabel("Simulation day")
    ax.set_ylabel("BESS SOC (%)")
    ax.set_title("Battery State of Charge Across the Seven-Day Operating Context")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(
        FIGURES / "17_7day_bess_soc.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)

    # Figure 3 — voltage envelope.
    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.fill_between(
        x_days,
        df["min_bus_v_pu"],
        df["max_bus_v_pu"],
        alpha=0.18,
        label="Bus-voltage envelope"
    )
    ax.plot(
        x_days,
        df["min_bus_v_pu"],
        linewidth=1.3,
        label="Minimum bus voltage"
    )
    ax.plot(
        x_days,
        df["max_bus_v_pu"],
        linewidth=1.3,
        label="Maximum bus voltage"
    )
    ax.set_xlabel("Simulation day")
    ax.set_ylabel("Voltage (pu)")
    ax.set_title("Seven-Day Bus-Voltage Operating Envelope")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(
        FIGURES / "18_7day_voltage_envelope.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)

    print("\n=== PHASE 2E RESEARCH-FIXED: CLEAN 7-DAY OPERATING CONTEXT ===")
    print(f"Samples                 : {len(df)}")
    print(f"Resolution              : {STEP_MIN} min")
    print(
        f"Power-flow success      : "
        f"{100*df['powerflow_ok'].mean():.2f}%"
    )
    print(
        f"PV energy               : "
        f"{summary.iloc[0]['pv_energy_mwh']:.3f} MWh"
    )
    print(
        f"Load energy             : "
        f"{summary.iloc[0]['load_energy_mwh']:.3f} MWh"
    )
    print(
        f"Min bus voltage         : "
        f"{summary.iloc[0]['min_bus_voltage_pu']:.4f} pu"
    )
    print(
        f"Max line loading        : "
        f"{summary.iloc[0]['max_line_loading_percent']:.2f}%"
    )
    print(
        f"BESS SOC range          : "
        f"{100*summary.iloc[0]['min_soc']:.1f}% .. "
        f"{100*summary.iloc[0]['max_soc']:.1f}%"
    )
    print(
        f"BESS mean SOC           : "
        f"{100*summary.iloc[0]['mean_soc']:.1f}%"
    )
    print(
        f"BESS SOC std            : "
        f"{100*summary.iloc[0]['soc_std']:.1f} percentage points"
    )
    print(
        f"Fast-event calendar rows: {len(event_calendar)}"
    )

    print("\nSaved:")
    print("  data/phase2e_7day_operating_dataset.csv")
    print("  data/phase2e_fast_event_calendar.csv")
    print("  results/phase2e_operating_summary.csv")
    print("  figures/16_7day_microgrid_operation.png")
    print("  figures/17_7day_bess_soc.png")
    print("  figures/18_7day_voltage_envelope.png")

    print(
        "\nNEXT:\n"
        "Phase 2F will use the event calendar to generate high-resolution "
        "GFL/GFM transient windows around selected physical, cyber and hybrid "
        "events. The slow and fast layers will then be linked into one "
        "multi-rate cyber-physical dataset."
    )


if __name__ == "__main__":
    run()
