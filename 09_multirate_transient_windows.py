from pathlib import Path
import importlib.util
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
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
inv = load_module("02_inverter_models.py", "inverter_models")


# ============================================================
# PHASE 2F — MATCHED HIGH-RESOLUTION TRANSIENT WINDOWS
# ============================================================
# Slow operating context:
#   7 days @ 5 min (from Phase 2E)
#
# Fast transient layer:
#   1 ms inverter-control integration
#   50 ms quasi-static AC-network refresh
#
# Every real/cyber/hybrid event is paired with a NORMAL counterfactual
# simulated from the SAME operating point.
# ============================================================

CONTROL_DT = 1e-3
NETWORK_DT = 50e-3
SIM_END = 3.0

EVENT_START = 1.0
EVENT_END = 1.50
CLOUD_END = 2.00

S_BASE_MVA = 0.10

GFL_IMAX_PU = 1.20
GFM_IMAX_PU = 1.20

PF_LOAD = 0.95


EVENT_FAMILY = {
    "physical_cloud_transient": "physical",
    "physical_load_spike": "physical",
    "physical_line_event": "physical",
    "cyber_fake_voltage_measurement": "cyber",
    "cyber_phase_drift": "cyber",
    "cyber_gfm_setpoint_manipulation": "cyber",
    "hybrid_fault_masking": "hybrid",
}


def reactive_from_pf(p_mw, pf=PF_LOAD):
    phi = np.arccos(pf)
    return float(p_mw * np.tan(phi))


def wrap_angle(rad):
    return (rad + np.pi) % (2*np.pi) - np.pi


def active_window(t, event_name):
    if event_name == "physical_cloud_transient":
        return EVENT_START <= t < CLOUD_END
    return EVENT_START <= t < EVENT_END


def collapse_event_calendar(df):
    """
    Phase 2E labels every 5-minute row inside an event.
    Collapse contiguous identical labels into one event block and retain
    the first slow operating point as the matched event context.
    """
    labels = df["event_label"].astype(str).to_numpy()
    events = []

    in_block = False
    start = None
    current_label = None

    for i, label in enumerate(labels):
        is_event = label != "normal"

        if is_event and (
            not in_block or label != current_label
        ):
            if in_block:
                events.append(
                    {
                        "event_label": current_label,
                        "start_row": start,
                        "end_row": i - 1,
                    }
                )

            in_block = True
            start = i
            current_label = label

        elif not is_event and in_block:
            events.append(
                {
                    "event_label": current_label,
                    "start_row": start,
                    "end_row": i - 1,
                }
            )
            in_block = False
            start = None
            current_label = None

    if in_block:
        events.append(
            {
                "event_label": current_label,
                "start_row": start,
                "end_row": len(df) - 1,
            }
        )

    out = pd.DataFrame(events)

    if out.empty:
        raise RuntimeError(
            "No event blocks found in Phase 2E dataset."
        )

    out["event_id"] = np.arange(len(out))
    return out


def cloud_multiplier(t):
    """
    Smooth cloud event:
      1.0 -> 0.35 over 0.25 s
      hold
      0.35 -> 1.0 before CLOUD_END
    """
    if t < EVENT_START or t >= CLOUD_END:
        return 1.0

    if t < EVENT_START + 0.25:
        u = (t - EVENT_START) / 0.25
        return 1.0 - 0.65*u

    if t < CLOUD_END - 0.25:
        return 0.35

    u = (t - (CLOUD_END - 0.25)) / 0.25
    return 0.35 + 0.65*u


def configure_physical_network(
    net,
    event_name,
    mode,
    t,
    load_a_base,
    load_b_base,
):
    """
    mode:
      'normal_counterfactual' -> no event
      'event'                 -> apply physical component where applicable
    """
    ext = net.ext_grid.index[0]

    net.ext_grid.at[ext, "vm_pu"] = 1.0
    net.ext_grid.at[ext, "va_degree"] = 0.0

    load_a = load_a_base
    load_b = load_b_base

    if mode == "event" and active_window(t, event_name):
        if event_name == "physical_load_spike":
            load_a = 1.55 * load_a_base

        elif event_name == "physical_line_event":
            # Reduced-order proxy for a severe upstream line/grid disturbance.
            # We do not open a radial branch because that would create an
            # unsupplied island in this small testbed.
            net.ext_grid.at[ext, "vm_pu"] = 0.72
            net.ext_grid.at[ext, "va_degree"] = 6.0

        elif event_name == "hybrid_fault_masking":
            # Real grid disturbance exists.
            net.ext_grid.at[ext, "vm_pu"] = 0.65
            net.ext_grid.at[ext, "va_degree"] = 7.0

    net.load.at[net.load.index[0], "p_mw"] = load_a
    net.load.at[
        net.load.index[0],
        "q_mvar"
    ] = reactive_from_pf(load_a)

    net.load.at[net.load.index[1], "p_mw"] = load_b
    net.load.at[
        net.load.index[1],
        "q_mvar"
    ] = reactive_from_pf(load_b)


def controller_channels(
    event_name,
    mode,
    t,
    v_true,
    theta_true,
    pv_pref_base,
    bess_pref_base,
):
    """
    Returns:
      GFL measured V/theta
      PV active-power reference
      BESS/GFM active-power reference
    """
    v_meas = v_true
    theta_meas = theta_true

    pv_pref = pv_pref_base
    bess_pref = bess_pref_base

    if mode != "event":
        return v_meas, theta_meas, pv_pref, bess_pref

    if event_name == "physical_cloud_transient":
        pv_pref = pv_pref_base * cloud_multiplier(t)

    elif event_name == "cyber_fake_voltage_measurement":
        if active_window(t, event_name):
            v_meas = 0.60
            theta_meas = np.deg2rad(8.0)

    elif event_name == "cyber_phase_drift":
        if active_window(t, event_name):
            attack_age = t - EVENT_START
            # Equivalent to a 0.45-Hz bias integrated into phase.
            theta_meas = (
                theta_true
                + 2*np.pi*0.45*attack_age
            )

    elif event_name == "cyber_gfm_setpoint_manipulation":
        if active_window(t, event_name):
            # Unauthorized positive dispatch shift.
            bess_pref = np.clip(
                bess_pref_base + 0.45,
                -0.80,
                0.80,
            )

    elif event_name == "hybrid_fault_masking":
        if active_window(t, event_name):
            # Real physical sag is hidden from the GFL controller.
            v_meas = 1.0
            theta_meas = 0.0

    return v_meas, theta_meas, pv_pref, bess_pref


def initialize_case(context_row):
    net = topology.build_microgrid()

    pv_idx = net.gridra["pv_sgen"]
    bess_idx = net.gridra["bess_sgen"]

    load_a = float(context_row["load_a_p_mw"])
    load_b = float(context_row["load_b_p_mw"])
    pv_p = float(context_row["pv_p_mw"])
    bess_p = float(context_row["bess_p_mw"])

    net.load.at[net.load.index[0], "p_mw"] = load_a
    net.load.at[
        net.load.index[0],
        "q_mvar"
    ] = reactive_from_pf(load_a)

    net.load.at[net.load.index[1], "p_mw"] = load_b
    net.load.at[
        net.load.index[1],
        "q_mvar"
    ] = reactive_from_pf(load_b)

    net.sgen.at[pv_idx, "p_mw"] = pv_p
    net.sgen.at[pv_idx, "q_mvar"] = 0.0

    net.sgen.at[bess_idx, "p_mw"] = bess_p
    net.sgen.at[bess_idx, "q_mvar"] = 0.0

    topology.solve_base_case(net)

    return net


def run_window(
    event_id,
    event_name,
    context_row,
    mode,
):
    net = initialize_case(context_row)

    pv_bus = net.gridra["bus_pv"]
    bess_bus = net.gridra["bus_bess"]
    pv_idx = net.gridra["pv_sgen"]
    bess_idx = net.gridra["bess_sgen"]

    pv_pref_base = (
        float(context_row["pv_p_mw"])
        / S_BASE_MVA
    )
    bess_pref_base = (
        float(context_row["bess_p_mw"])
        / S_BASE_MVA
    )

    load_a_base = float(context_row["load_a_p_mw"])
    load_b_base = float(context_row["load_b_p_mw"])
    soc_context = float(context_row["bess_soc"])

    gfl = inv.GFLInverter(
        kp_pll=18.0,
        ki_pll=250.0,
        i_max_pu=GFL_IMAX_PU,
        q_voltage_gain=2.0,
    )

    gfm = inv.GFMInverter(
        m_inertia=0.25,
        d_damping=0.80,
        e_internal_pu=1.02,
        x_virtual_pu=0.25,
        i_max_pu=GFM_IMAX_PU,
    )

    theta_pv_0 = np.deg2rad(
        float(net.res_bus.at[pv_bus, "va_degree"])
    )
    theta_bess_0 = np.deg2rad(
        float(net.res_bus.at[bess_bus, "va_degree"])
    )

    v_bess_0 = float(
        net.res_bus.at[bess_bus, "vm_pu"]
    )

    gfl.reset(theta0=theta_pv_0)

    # Approximate GFM equilibrium around the operating point.
    arg = (
        bess_pref_base
        * gfm.x_virtual_pu
        / max(gfm.e_internal_pu * v_bess_0, 1e-6)
    )
    delta0 = np.arcsin(
        np.clip(arg, -0.90, 0.90)
    )
    gfm.reset(theta0=theta_bess_0 + delta0)

    pv_p_mw = float(context_row["pv_p_mw"])
    pv_q_mvar = 0.0

    bess_p_mw = float(context_row["bess_p_mw"])
    bess_q_mvar = 0.0

    v_pv_true = float(
        net.res_bus.at[pv_bus, "vm_pu"]
    )
    theta_pv_true = theta_pv_0

    v_bess_true = float(
        net.res_bus.at[bess_bus, "vm_pu"]
    )
    theta_bess_true = theta_bess_0

    min_bus_v = float(net.res_bus["vm_pu"].min())
    max_line_loading = float(
        net.res_line["loading_percent"].max()
    )
    ext_p_mw = float(net.res_ext_grid.iloc[0]["p_mw"])

    network_stride = max(
        1,
        int(round(NETWORK_DT / CONTROL_DT))
    )

    rows = []

    times = np.arange(
        0.0,
        SIM_END + CONTROL_DT/2,
        CONTROL_DT,
    )

    for k, t in enumerate(times):
        # --------------------------------------------------------
        # 1. Slow/physical network refresh
        # --------------------------------------------------------
        if k % network_stride == 0:
            configure_physical_network(
                net,
                event_name,
                mode,
                t,
                load_a_base,
                load_b_base,
            )

            net.sgen.at[pv_idx, "p_mw"] = pv_p_mw
            net.sgen.at[pv_idx, "q_mvar"] = pv_q_mvar

            net.sgen.at[bess_idx, "p_mw"] = bess_p_mw
            net.sgen.at[bess_idx, "q_mvar"] = bess_q_mvar

            topology.solve_base_case(net)

            v_pv_true = float(
                net.res_bus.at[pv_bus, "vm_pu"]
            )
            theta_pv_true = np.deg2rad(
                float(
                    net.res_bus.at[
                        pv_bus,
                        "va_degree"
                    ]
                )
            )

            v_bess_true = float(
                net.res_bus.at[bess_bus, "vm_pu"]
            )
            theta_bess_true = np.deg2rad(
                float(
                    net.res_bus.at[
                        bess_bus,
                        "va_degree"
                    ]
                )
            )

            min_bus_v = float(
                net.res_bus["vm_pu"].min()
            )
            max_line_loading = float(
                net.res_line[
                    "loading_percent"
                ].max()
            )
            ext_p_mw = float(
                net.res_ext_grid.iloc[0]["p_mw"]
            )

        # --------------------------------------------------------
        # 2. Controller / cyber channels
        # --------------------------------------------------------
        (
            v_gfl_meas,
            theta_gfl_meas,
            pv_pref,
            bess_pref,
        ) = controller_channels(
            event_name,
            mode,
            t,
            v_pv_true,
            theta_pv_true,
            pv_pref_base,
            bess_pref_base,
        )

        # --------------------------------------------------------
        # 3. Fast inverter dynamics
        # --------------------------------------------------------
        gfl_out = gfl.step(
            v_meas_pu=v_gfl_meas,
            theta_meas_rad=theta_gfl_meas,
            p_ref_pu=pv_pref,
            q_ref_pu=0.0,
            dt=CONTROL_DT,
        )

        gfm_out = gfm.step(
            v_pcc_pu=v_bess_true,
            theta_pcc_rad=theta_bess_true,
            p_ref_pu=bess_pref,
            dt=CONTROL_DT,
        )

        # --------------------------------------------------------
        # 4. Close loop back to AC network
        # --------------------------------------------------------
        pv_p_mw = (
            gfl_out["p_out_pu"]
            * S_BASE_MVA
        )

        # Convert GFL internal sign convention to positive
        # reactive injection into the network.
        pv_q_mvar = (
            -gfl_out["q_out_pu"]
            * S_BASE_MVA
        )

        bess_p_mw = (
            gfm_out["p_out_pu"]
            * S_BASE_MVA
        )
        bess_q_mvar = (
            gfm_out["q_out_pu"]
            * S_BASE_MVA
        )

        # --------------------------------------------------------
        # 5. Cyber-physical feature layer
        # --------------------------------------------------------
        sensor_v_residual = (
            v_gfl_meas - v_pv_true
        )
        sensor_angle_residual_deg = np.rad2deg(
            wrap_angle(
                theta_gfl_meas
                - theta_pv_true
            )
        )

        rows.append(
            {
                "event_id": event_id,
                "event_label": event_name,
                "event_family": EVENT_FAMILY[
                    event_name
                ],
                "window_mode": mode,
                "context_timestamp": context_row[
                    "timestamp"
                ],
                "context_pv_p_mw": float(
                    context_row["pv_p_mw"]
                ),
                "context_load_total_p_mw": float(
                    context_row[
                        "load_total_p_mw"
                    ]
                ),
                "context_bess_soc": soc_context,

                "time_s": t,
                "event_active": bool(
                    active_window(t, event_name)
                    and mode == "event"
                ),

                "v_pv_true_pu": v_pv_true,
                "v_gfl_measured_pu": v_gfl_meas,
                "sensor_v_residual_pu": sensor_v_residual,

                "theta_pv_true_deg": np.rad2deg(
                    theta_pv_true
                ),
                "theta_gfl_measured_deg": np.rad2deg(
                    theta_gfl_meas
                ),
                "sensor_angle_residual_deg": (
                    sensor_angle_residual_deg
                ),

                "min_bus_v_pu": min_bus_v,
                "max_line_loading_percent": (
                    max_line_loading
                ),
                "ext_grid_p_mw": ext_p_mw,

                "pv_pref_pu": pv_pref,
                "gfl_pll_angle_deg": (
                    gfl_out["theta_pll_deg"]
                ),
                "gfl_pll_freq_dev_hz": (
                    gfl_out[
                        "pll_freq_dev_hz"
                    ]
                ),
                "gfl_id_pu": gfl_out["id_pu"],
                "gfl_iq_pu": gfl_out["iq_pu"],
                "gfl_current_pu": (
                    gfl_out["i_mag_pu"]
                ),
                "gfl_limiter": bool(
                    gfl_out["limiter_active"]
                ),

                "bess_pref_pu": bess_pref,
                "gfm_angle_deg": (
                    gfm_out[
                        "theta_internal_deg"
                    ]
                ),
                "gfm_freq_dev_hz": (
                    gfm_out[
                        "gfm_freq_dev_hz"
                    ]
                ),
                "gfm_p_pu": gfm_out["p_out_pu"],
                "gfm_q_pu": gfm_out["q_out_pu"],
                "gfm_current_pu": (
                    gfm_out["i_mag_pu"]
                ),
                "gfm_limiter": bool(
                    gfm_out["limiter_active"]
                ),
            }
        )

    return pd.DataFrame(rows)


def summarize_window(df):
    event = df[df["event_active"]]

    # Normal counterfactual has no event_active samples;
    # use the corresponding event time interval for fair metrics.
    if event.empty:
        if (
            df.iloc[0]["event_label"]
            == "physical_cloud_transient"
        ):
            event = df[
                (df["time_s"] >= EVENT_START)
                & (df["time_s"] < CLOUD_END)
            ]
        else:
            event = df[
                (df["time_s"] >= EVENT_START)
                & (df["time_s"] < EVENT_END)
            ]

    return {
        "event_id": int(df.iloc[0]["event_id"]),
        "event_label": str(
            df.iloc[0]["event_label"]
        ),
        "event_family": str(
            df.iloc[0]["event_family"]
        ),
        "window_mode": str(
            df.iloc[0]["window_mode"]
        ),
        "context_timestamp": df.iloc[0][
            "context_timestamp"
        ],
        "context_pv_p_mw": float(
            df.iloc[0]["context_pv_p_mw"]
        ),
        "context_load_total_p_mw": float(
            df.iloc[0][
                "context_load_total_p_mw"
            ]
        ),
        "context_bess_soc": float(
            df.iloc[0]["context_bess_soc"]
        ),

        "min_bus_v_pu": float(
            event["min_bus_v_pu"].min()
        ),
        "max_line_loading_percent": float(
            event[
                "max_line_loading_percent"
            ].max()
        ),
        "max_abs_sensor_v_residual_pu": float(
            event[
                "sensor_v_residual_pu"
            ].abs().max()
        ),
        "max_abs_sensor_angle_residual_deg": float(
            event[
                "sensor_angle_residual_deg"
            ].abs().max()
        ),
        "max_abs_gfl_pll_freq_hz": float(
            event[
                "gfl_pll_freq_dev_hz"
            ].abs().max()
        ),
        "max_abs_gfm_freq_hz": float(
            event[
                "gfm_freq_dev_hz"
            ].abs().max()
        ),
        "max_gfl_current_pu": float(
            event["gfl_current_pu"].max()
        ),
        "max_gfm_current_pu": float(
            event["gfm_current_pu"].max()
        ),
        "gfl_limiter_fraction": float(
            event["gfl_limiter"].mean()
        ),
        "gfm_limiter_fraction": float(
            event["gfm_limiter"].mean()
        ),
    }


def add_matched_deltas(summary):
    out = summary.copy()

    metrics = [
        "min_bus_v_pu",
        "max_line_loading_percent",
        "max_abs_sensor_v_residual_pu",
        "max_abs_sensor_angle_residual_deg",
        "max_abs_gfl_pll_freq_hz",
        "max_abs_gfm_freq_hz",
        "max_gfl_current_pu",
        "max_gfm_current_pu",
        "gfl_limiter_fraction",
        "gfm_limiter_fraction",
    ]

    for metric in metrics:
        out[f"delta_{metric}"] = np.nan

    for event_id in out["event_id"].unique():
        normal = out[
            (out["event_id"] == event_id)
            & (
                out["window_mode"]
                == "normal_counterfactual"
            )
        ]
        event = out[
            (out["event_id"] == event_id)
            & (out["window_mode"] == "event")
        ]

        if normal.empty or event.empty:
            continue

        nidx = normal.index[0]
        eidx = event.index[0]

        for metric in metrics:
            baseline = float(
                out.at[nidx, metric]
            )
            event_value = float(
                out.at[eidx, metric]
            )

            out.at[
                eidx,
                f"delta_{metric}"
            ] = event_value - baseline

            out.at[
                nidx,
                f"delta_{metric}"
            ] = 0.0

    return out


def plot_event_fingerprint(summary):
    d = summary[
        summary["window_mode"] == "event"
    ].copy()

    x = d[
        "max_abs_gfl_pll_freq_hz"
    ].to_numpy()
    y = d[
        "max_abs_gfm_freq_hz"
    ].to_numpy()

    fig, ax = plt.subplots(
        figsize=(9.5, 6.7)
    )

    ax.scatter(x, y, s=90)

    for _, row in d.iterrows():
        ax.annotate(
            row["event_label"],
            (
                row[
                    "max_abs_gfl_pll_freq_hz"
                ],
                row[
                    "max_abs_gfm_freq_hz"
                ],
            ),
            xytext=(6, 5),
            textcoords="offset points",
            fontsize=8.5,
        )

    ax.set_xlabel(
        "Maximum |GFL PLL frequency deviation| (Hz)"
    )
    ax.set_ylabel(
        "Maximum |GFM frequency deviation| (Hz)"
    )
    ax.set_title(
        "Matched Inverter-Dynamic Fingerprints Across Physical, Cyber and Hybrid Events"
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(
        FIGURES
        / "19_multirate_event_fingerprints.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_matched_voltage_effect(summary):
    d = summary[
        summary["window_mode"] == "event"
    ].copy()

    labels = d["event_label"].tolist()
    vals = (
        100
        * d[
            "delta_min_bus_v_pu"
        ].to_numpy()
    )

    fig, ax = plt.subplots(
        figsize=(11, 6)
    )

    x = np.arange(len(labels))
    ax.bar(x, vals)

    ax.axhline(0.0, linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(
        labels,
        rotation=25,
        ha="right",
    )
    ax.set_ylabel(
        "Matched change in minimum bus voltage (% of 1 pu)"
    )
    ax.set_title(
        "Event Effect Relative to the Same Operating-Point Counterfactual"
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(
        FIGURES
        / "20_matched_counterfactual_voltage_effect.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_hybrid_masking(timeseries):
    d = timeseries[
        (
            timeseries["event_label"]
            == "hybrid_fault_masking"
        )
        & (
            timeseries["window_mode"]
            == "event"
        )
    ]

    if d.empty:
        return

    fig, ax = plt.subplots(
        figsize=(10, 5.8)
    )

    ax.plot(
        d["time_s"],
        d["v_pv_true_pu"],
        linewidth=2.2,
        label="Physical PV-bus voltage",
    )
    ax.plot(
        d["time_s"],
        d["v_gfl_measured_pu"],
        linewidth=2.2,
        linestyle="--",
        label="Compromised GFL measurement",
    )

    ax.axvspan(
        EVENT_START,
        EVENT_END,
        alpha=0.10,
    )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Voltage (pu)")
    ax.set_title(
        "Hybrid Fault Masking in the Multi-Rate Digital Twin"
    )
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(
        FIGURES
        / "21_multirate_hybrid_fault_masking.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)


if __name__ == "__main__":
    slow_path = (
        DATA
        / "phase2e_7day_operating_dataset.csv"
    )

    if not slow_path.exists():
        raise FileNotFoundError(
            "Missing Phase 2E dataset: "
            + str(slow_path)
        )

    slow = pd.read_csv(
        slow_path,
        parse_dates=["timestamp"],
    )

    event_blocks = collapse_event_calendar(
        slow
    )

    all_windows = []
    summary_rows = []

    print(
        "\n=== PHASE 2F: MATCHED HIGH-RESOLUTION TRANSIENT WINDOWS ==="
    )
    print(
        f"Event blocks found: {len(event_blocks)}"
    )
    print(
        f"Fast integration step: {CONTROL_DT*1000:.1f} ms"
    )
    print(
        f"AC-network refresh: {NETWORK_DT*1000:.0f} ms"
    )

    for _, block in event_blocks.iterrows():
        event_id = int(block["event_id"])
        event_name = str(
            block["event_label"]
        )

        if event_name not in EVENT_FAMILY:
            print(
                f"Skipping unknown event: {event_name}"
            )
            continue

        context = slow.iloc[
            int(block["start_row"])
        ]

        print(
            f"\nEvent {event_id}: {event_name}"
        )
        print(
            f"  context={context['timestamp']} | "
            f"PV={1000*context['pv_p_mw']:.1f} kW | "
            f"Load={1000*context['load_total_p_mw']:.1f} kW | "
            f"SOC={100*context['bess_soc']:.1f}%"
        )

        for mode in [
            "normal_counterfactual",
            "event",
        ]:
            print(
                f"  running {mode} ..."
            )

            df = run_window(
                event_id,
                event_name,
                context,
                mode,
            )

            all_windows.append(df)
            summary_rows.append(
                summarize_window(df)
            )

    timeseries = pd.concat(
        all_windows,
        ignore_index=True,
    )
    summary = pd.DataFrame(
        summary_rows
    )

    summary = add_matched_deltas(
        summary
    )

    timeseries.to_csv(
        DATA
        / "phase2f_multirate_transient_windows.csv",
        index=False,
    )

    summary.to_csv(
        RESULTS
        / "phase2f_matched_window_summary.csv",
        index=False,
    )

    plot_event_fingerprint(summary)
    plot_matched_voltage_effect(summary)
    plot_hybrid_masking(timeseries)

    print("\n=== MATCHED EVENT SUMMARY ===")

    show = summary[
        summary["window_mode"] == "event"
    ][
        [
            "event_id",
            "event_label",
            "event_family",
            "context_pv_p_mw",
            "context_load_total_p_mw",
            "context_bess_soc",
            "min_bus_v_pu",
            "max_abs_gfl_pll_freq_hz",
            "max_abs_gfm_freq_hz",
            "max_gfl_current_pu",
            "max_gfm_current_pu",
            "delta_min_bus_v_pu",
        ]
    ]

    print(
        show.to_string(index=False)
    )

    print("\nSaved:")
    print(
        "  data/phase2f_multirate_transient_windows.csv"
    )
    print(
        "  results/phase2f_matched_window_summary.csv"
    )
    print(
        "  figures/19_multirate_event_fingerprints.png"
    )
    print(
        "  figures/20_matched_counterfactual_voltage_effect.png"
    )
    print(
        "  figures/21_multirate_hybrid_fault_masking.png"
    )

    print(
        "\nNEXT:\n"
        "Phase 2G will turn these matched windows into a feature dataset "
        "with temporal statistics, WLS state-estimation residual features "
        "and graph/topology descriptors. That dataset will be the first "
        "proper input to the final detection/localization model."
    )
