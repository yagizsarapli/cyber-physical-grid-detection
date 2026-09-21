from pathlib import Path
import importlib.util
import argparse
import copy
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
    spec = importlib.util.spec_from_file_location(
        module_name,
        ROOT / filename
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


topology = load_module(
    "01_microgrid_topology.py",
    "topology"
)
inv = load_module(
    "02_inverter_models.py",
    "inverter_models"
)


# ============================================================
# PHASE 2G — RANDOMIZED MATCHED CYBER-PHYSICAL BENCHMARK
# ============================================================
#
# Purpose
# -------
# Replace the 7 hand-picked demonstration events with a reproducible
# population of randomized scenarios sampled across:
#
#   - operating point (PV, load, SOC, time of day)
#   - event severity
#   - event duration
#   - event start time
#   - line-parameter uncertainty
#   - controller-parameter uncertainty
#   - sensor noise
#
# Every abnormal scenario has a matched NORMAL counterfactual using:
#
#   SAME operating point
#   SAME network parameters
#   SAME controller parameters
#   SAME sensor-noise realization
#
# so event-vs-normal differences are not confounded by context.
# ============================================================

S_BASE_MVA = 0.10

CONTROL_DT = 1e-3
NETWORK_DT = 50e-3
PRE_EVENT_MIN = 0.65
POST_EVENT_RECOVERY = 0.85
MAX_SIM_END = 2.60

BASE_GFL_IMAX = 1.20
BASE_GFM_IMAX = 1.20

PF_LOAD = 0.95

EVENT_TYPES = [
    "physical_cloud_transient",
    "physical_load_spike",
    "physical_voltage_sag",
    "cyber_fake_voltage_measurement",
    "cyber_phase_drift",
    "cyber_gfl_current_limit",
    "cyber_gfm_setpoint_manipulation",
    "hybrid_fault_masking",
]

EVENT_FAMILY = {
    "physical_cloud_transient": "physical",
    "physical_load_spike": "physical",
    "physical_voltage_sag": "physical",
    "cyber_fake_voltage_measurement": "cyber",
    "cyber_phase_drift": "cyber",
    "cyber_gfl_current_limit": "cyber",
    "cyber_gfm_setpoint_manipulation": "cyber",
    "hybrid_fault_masking": "hybrid",
}


def reactive_from_pf(p_mw, pf=PF_LOAD):
    phi = np.arccos(pf)
    return float(p_mw * np.tan(phi))


def wrap_angle(rad):
    return (rad + np.pi) % (2*np.pi) - np.pi


def sample_scenario_params(rng, event_type):
    start = float(rng.uniform(0.70, 0.95))
    duration = float(rng.uniform(0.25, 0.75))
    end = start + duration
    sim_end = min(
        MAX_SIM_END,
        end + POST_EVENT_RECOVERY
    )

    p = {
        "event_start_s": start,
        "event_duration_s": duration,
        "event_end_s": end,
        "sim_end_s": sim_end,

        # Parameter uncertainty.
        "line_r_scale": float(rng.uniform(0.90, 1.10)),
        "line_x_scale": float(rng.uniform(0.90, 1.10)),

        "gfl_kp_scale": float(rng.uniform(0.90, 1.10)),
        "gfl_ki_scale": float(rng.uniform(0.90, 1.10)),
        "gfm_m_scale": float(rng.uniform(0.90, 1.10)),
        "gfm_d_scale": float(rng.uniform(0.90, 1.10)),

        # Measurement uncertainty.
        "v_noise_sigma_pu": float(rng.uniform(0.0005, 0.0025)),
        "angle_noise_sigma_deg": float(rng.uniform(0.01, 0.06)),

        # Default event-specific fields.
        "cloud_min_factor": 1.0,
        "cloud_ramp_s": 0.15,
        "load_spike_factor": 1.0,

        "physical_sag_vm_pu": 1.0,
        "physical_angle_jump_deg": 0.0,

        "fake_vm_pu": 1.0,
        "fake_angle_deg": 0.0,

        "phase_drift_hz": 0.0,

        "gfl_attack_imax_pu": BASE_GFL_IMAX,
        "gfm_pref_delta_pu": 0.0,

        "mask_vm_pu": 1.0,
        "mask_angle_deg": 0.0,
    }

    if event_type == "physical_cloud_transient":
        p["cloud_min_factor"] = float(
            rng.uniform(0.25, 0.70)
        )
        p["cloud_ramp_s"] = float(
            rng.uniform(0.08, 0.20)
        )

    elif event_type == "physical_load_spike":
        p["load_spike_factor"] = float(
            rng.uniform(1.20, 1.80)
        )

    elif event_type == "physical_voltage_sag":
        p["physical_sag_vm_pu"] = float(
            rng.uniform(0.55, 0.85)
        )
        p["physical_angle_jump_deg"] = float(
            rng.uniform(2.0, 10.0)
        )

    elif event_type == "cyber_fake_voltage_measurement":
        p["fake_vm_pu"] = float(
            rng.uniform(0.55, 0.90)
        )
        p["fake_angle_deg"] = float(
            rng.uniform(2.0, 10.0)
        )

    elif event_type == "cyber_phase_drift":
        p["phase_drift_hz"] = float(
            rng.uniform(0.10, 0.70)
        )

    elif event_type == "cyber_gfl_current_limit":
        p["gfl_attack_imax_pu"] = float(
            rng.uniform(0.20, 0.75)
        )

    elif event_type == "cyber_gfm_setpoint_manipulation":
        sign = rng.choice([-1.0, 1.0])
        p["gfm_pref_delta_pu"] = float(
            sign * rng.uniform(0.15, 0.50)
        )

    elif event_type == "hybrid_fault_masking":
        p["physical_sag_vm_pu"] = float(
            rng.uniform(0.55, 0.80)
        )
        p["physical_angle_jump_deg"] = float(
            rng.uniform(3.0, 10.0)
        )
        p["mask_vm_pu"] = float(
            rng.uniform(0.97, 1.02)
        )
        p["mask_angle_deg"] = float(
            rng.uniform(-0.5, 0.5)
        )

    return p


def is_event_active(t, p):
    return (
        p["event_start_s"]
        <= t
        < p["event_end_s"]
    )


def cloud_factor(t, p):
    if not is_event_active(t, p):
        return 1.0

    start = p["event_start_s"]
    end = p["event_end_s"]
    ramp = min(
        p["cloud_ramp_s"],
        0.45 * p["event_duration_s"]
    )
    minimum = p["cloud_min_factor"]

    if t < start + ramp:
        u = (t - start) / max(ramp, 1e-6)
        return 1.0 - (1.0 - minimum) * u

    if t >= end - ramp:
        u = (t - (end - ramp)) / max(ramp, 1e-6)
        return minimum + (1.0 - minimum) * u

    return minimum


def configure_network(
    net,
    event_type,
    mode,
    t,
    p,
    load_a_base,
    load_b_base,
):
    ext = net.ext_grid.index[0]

    net.ext_grid.at[ext, "vm_pu"] = 1.0
    net.ext_grid.at[ext, "va_degree"] = 0.0

    load_a = load_a_base
    load_b = load_b_base

    if mode == "event" and is_event_active(t, p):
        if event_type == "physical_load_spike":
            load_a = (
                p["load_spike_factor"]
                * load_a_base
            )

        elif event_type in (
            "physical_voltage_sag",
            "hybrid_fault_masking",
        ):
            net.ext_grid.at[
                ext,
                "vm_pu"
            ] = p["physical_sag_vm_pu"]

            net.ext_grid.at[
                ext,
                "va_degree"
            ] = p["physical_angle_jump_deg"]

    net.load.at[
        net.load.index[0],
        "p_mw"
    ] = load_a
    net.load.at[
        net.load.index[0],
        "q_mvar"
    ] = reactive_from_pf(load_a)

    net.load.at[
        net.load.index[1],
        "p_mw"
    ] = load_b
    net.load.at[
        net.load.index[1],
        "q_mvar"
    ] = reactive_from_pf(load_b)


def controller_channels(
    event_type,
    mode,
    t,
    p,
    v_true,
    theta_true,
    pv_pref_base,
    bess_pref_base,
    v_noise,
    theta_noise_rad,
):
    # Same measurement-noise realization is used for event and counterfactual.
    v_meas = v_true + v_noise
    theta_meas = theta_true + theta_noise_rad

    pv_pref = pv_pref_base
    bess_pref = bess_pref_base
    gfl_imax = BASE_GFL_IMAX

    if mode != "event":
        return (
            v_meas,
            theta_meas,
            pv_pref,
            bess_pref,
            gfl_imax,
        )

    active = is_event_active(t, p)

    if event_type == "physical_cloud_transient":
        pv_pref = (
            pv_pref_base
            * cloud_factor(t, p)
        )

    elif (
        event_type
        == "cyber_fake_voltage_measurement"
        and active
    ):
        v_meas = p["fake_vm_pu"] + v_noise
        theta_meas = (
            np.deg2rad(p["fake_angle_deg"])
            + theta_noise_rad
        )

    elif (
        event_type
        == "cyber_phase_drift"
        and active
    ):
        age = t - p["event_start_s"]
        theta_meas = (
            theta_true
            + 2*np.pi*p["phase_drift_hz"]*age
            + theta_noise_rad
        )

    elif (
        event_type
        == "cyber_gfl_current_limit"
        and active
    ):
        gfl_imax = p["gfl_attack_imax_pu"]

    elif (
        event_type
        == "cyber_gfm_setpoint_manipulation"
        and active
    ):
        bess_pref = np.clip(
            bess_pref_base
            + p["gfm_pref_delta_pu"],
            -0.80,
            0.80,
        )

    elif (
        event_type
        == "hybrid_fault_masking"
        and active
    ):
        v_meas = p["mask_vm_pu"] + v_noise
        theta_meas = (
            np.deg2rad(p["mask_angle_deg"])
            + theta_noise_rad
        )

    return (
        v_meas,
        theta_meas,
        pv_pref,
        bess_pref,
        gfl_imax,
    )


def initialize_case(context, p):
    net = topology.build_microgrid()

    # Apply modest network-parameter uncertainty.
    net.line.loc[:, "r_ohm_per_km"] *= p["line_r_scale"]
    net.line.loc[:, "x_ohm_per_km"] *= p["line_x_scale"]

    pv_idx = net.gridra["pv_sgen"]
    bess_idx = net.gridra["bess_sgen"]

    load_a = float(context["load_a_p_mw"])
    load_b = float(context["load_b_p_mw"])
    pv_p = float(context["pv_p_mw"])
    bess_p = float(context["bess_p_mw"])

    net.load.at[
        net.load.index[0],
        "p_mw"
    ] = load_a
    net.load.at[
        net.load.index[0],
        "q_mvar"
    ] = reactive_from_pf(load_a)

    net.load.at[
        net.load.index[1],
        "p_mw"
    ] = load_b
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
    scenario_id,
    event_type,
    context,
    p,
    mode,
    noise_v,
    noise_theta_rad,
):
    net = initialize_case(context, p)

    pv_bus = net.gridra["bus_pv"]
    bess_bus = net.gridra["bus_bess"]

    pv_idx = net.gridra["pv_sgen"]
    bess_idx = net.gridra["bess_sgen"]

    pv_pref_base = (
        float(context["pv_p_mw"])
        / S_BASE_MVA
    )
    bess_pref_base = (
        float(context["bess_p_mw"])
        / S_BASE_MVA
    )

    load_a_base = float(context["load_a_p_mw"])
    load_b_base = float(context["load_b_p_mw"])

    gfl = inv.GFLInverter(
        kp_pll=18.0 * p["gfl_kp_scale"],
        ki_pll=250.0 * p["gfl_ki_scale"],
        i_max_pu=BASE_GFL_IMAX,
        q_voltage_gain=2.0,
    )

    gfm = inv.GFMInverter(
        m_inertia=0.25 * p["gfm_m_scale"],
        d_damping=0.80 * p["gfm_d_scale"],
        e_internal_pu=1.02,
        x_virtual_pu=0.25,
        i_max_pu=BASE_GFM_IMAX,
    )

    theta_pv_0 = np.deg2rad(
        float(
            net.res_bus.at[
                pv_bus,
                "va_degree"
            ]
        )
    )
    theta_bess_0 = np.deg2rad(
        float(
            net.res_bus.at[
                bess_bus,
                "va_degree"
            ]
        )
    )

    v_bess_0 = float(
        net.res_bus.at[
            bess_bus,
            "vm_pu"
        ]
    )

    gfl.reset(theta0=theta_pv_0)

    arg = (
        bess_pref_base
        * gfm.x_virtual_pu
        / max(
            gfm.e_internal_pu * v_bess_0,
            1e-6
        )
    )

    delta0 = np.arcsin(
        np.clip(arg, -0.90, 0.90)
    )

    gfm.reset(
        theta0=theta_bess_0 + delta0
    )

    pv_p_mw = float(context["pv_p_mw"])
    pv_q_mvar = 0.0

    bess_p_mw = float(context["bess_p_mw"])
    bess_q_mvar = 0.0

    v_pv_true = float(
        net.res_bus.at[pv_bus, "vm_pu"]
    )
    theta_pv_true = theta_pv_0

    v_bess_true = float(
        net.res_bus.at[bess_bus, "vm_pu"]
    )
    theta_bess_true = theta_bess_0

    min_bus_v = float(
        net.res_bus["vm_pu"].min()
    )
    max_line_loading = float(
        net.res_line[
            "loading_percent"
        ].max()
    )

    network_stride = max(
        1,
        int(round(NETWORK_DT / CONTROL_DT))
    )

    n_steps = int(
        np.floor(
            p["sim_end_s"] / CONTROL_DT
        )
    ) + 1

    times = (
        np.arange(n_steps, dtype=float)
        * CONTROL_DT
    )

    rows = []

    for k, t in enumerate(times):
        if k % network_stride == 0:
            configure_network(
                net,
                event_type,
                mode,
                t,
                p,
                load_a_base,
                load_b_base,
            )

            net.sgen.at[
                pv_idx,
                "p_mw"
            ] = pv_p_mw
            net.sgen.at[
                pv_idx,
                "q_mvar"
            ] = pv_q_mvar

            net.sgen.at[
                bess_idx,
                "p_mw"
            ] = bess_p_mw
            net.sgen.at[
                bess_idx,
                "q_mvar"
            ] = bess_q_mvar

            topology.solve_base_case(net)

            v_pv_true = float(
                net.res_bus.at[
                    pv_bus,
                    "vm_pu"
                ]
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
                net.res_bus.at[
                    bess_bus,
                    "vm_pu"
                ]
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

        (
            v_gfl_meas,
            theta_gfl_meas,
            pv_pref,
            bess_pref,
            gfl_imax,
        ) = controller_channels(
            event_type,
            mode,
            t,
            p,
            v_pv_true,
            theta_pv_true,
            pv_pref_base,
            bess_pref_base,
            noise_v[k],
            noise_theta_rad[k],
        )

        # Dynamic current-limit parameter channel.
        gfl.i_max_pu = gfl_imax

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

        pv_p_mw = (
            gfl_out["p_out_pu"]
            * S_BASE_MVA
        )
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

        sensor_v_residual = (
            v_gfl_meas - v_pv_true
        )
        sensor_angle_residual_deg = (
            np.rad2deg(
                wrap_angle(
                    theta_gfl_meas
                    - theta_pv_true
                )
            )
        )

        rows.append({
            "scenario_id": scenario_id,
            "event_type": event_type,
            "event_family": EVENT_FAMILY[
                event_type
            ],
            "window_mode": mode,

            "context_timestamp": context[
                "timestamp"
            ],
            "context_pv_p_mw": float(
                context["pv_p_mw"]
            ),
            "context_load_total_p_mw": float(
                context["load_total_p_mw"]
            ),
            "context_bess_soc": float(
                context["bess_soc"]
            ),

            "time_s": t,
            "event_active": bool(
                mode == "event"
                and is_event_active(t, p)
            ),

            "min_bus_v_pu": min_bus_v,
            "max_line_loading_percent": (
                max_line_loading
            ),

            "v_pv_true_pu": v_pv_true,
            "v_gfl_measured_pu": v_gfl_meas,
            "sensor_v_residual_pu": (
                sensor_v_residual
            ),

            "theta_pv_true_deg": np.rad2deg(
                theta_pv_true
            ),
            "theta_gfl_measured_deg": (
                np.rad2deg(theta_gfl_meas)
            ),
            "sensor_angle_residual_deg": (
                sensor_angle_residual_deg
            ),

            "pv_pref_pu": pv_pref,
            "bess_pref_pu": bess_pref,
            "gfl_imax_pu": gfl_imax,

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
        })

    return pd.DataFrame(rows)


def metric_window(df, p):
    return df[
        (df["time_s"] >= p["event_start_s"])
        & (
            df["time_s"]
            < p["event_end_s"]
        )
    ]


def summarize(df, p):
    d = metric_window(df, p)

    return {
        "scenario_id": int(
            df.iloc[0]["scenario_id"]
        ),
        "event_type": str(
            df.iloc[0]["event_type"]
        ),
        "event_family": str(
            df.iloc[0]["event_family"]
        ),
        "window_mode": str(
            df.iloc[0]["window_mode"]
        ),

        "context_timestamp": (
            df.iloc[0]["context_timestamp"]
        ),
        "context_pv_p_mw": float(
            df.iloc[0][
                "context_pv_p_mw"
            ]
        ),
        "context_load_total_p_mw": float(
            df.iloc[0][
                "context_load_total_p_mw"
            ]
        ),
        "context_bess_soc": float(
            df.iloc[0][
                "context_bess_soc"
            ]
        ),

        "min_bus_v_pu": float(
            d["min_bus_v_pu"].min()
        ),
        "max_line_loading_percent": float(
            d[
                "max_line_loading_percent"
            ].max()
        ),

        "max_abs_sensor_v_residual_pu": float(
            d[
                "sensor_v_residual_pu"
            ].abs().max()
        ),
        "rms_sensor_v_residual_pu": float(
            np.sqrt(
                np.mean(
                    d[
                        "sensor_v_residual_pu"
                    ].to_numpy() ** 2
                )
            )
        ),

        "max_abs_sensor_angle_residual_deg": float(
            d[
                "sensor_angle_residual_deg"
            ].abs().max()
        ),

        "max_abs_gfl_pll_freq_hz": float(
            d[
                "gfl_pll_freq_dev_hz"
            ].abs().max()
        ),
        "rms_gfl_pll_freq_hz": float(
            np.sqrt(
                np.mean(
                    d[
                        "gfl_pll_freq_dev_hz"
                    ].to_numpy() ** 2
                )
            )
        ),

        "max_abs_gfm_freq_hz": float(
            d[
                "gfm_freq_dev_hz"
            ].abs().max()
        ),

        "max_gfl_current_pu": float(
            d["gfl_current_pu"].max()
        ),
        "max_gfm_current_pu": float(
            d["gfm_current_pu"].max()
        ),

        "gfl_limiter_fraction": float(
            d["gfl_limiter"].mean()
        ),
        "gfm_limiter_fraction": float(
            d["gfm_limiter"].mean()
        ),
    }


def add_matched_deltas(summary):
    result = summary.copy()

    metrics = [
        "min_bus_v_pu",
        "max_line_loading_percent",
        "max_abs_sensor_v_residual_pu",
        "rms_sensor_v_residual_pu",
        "max_abs_sensor_angle_residual_deg",
        "max_abs_gfl_pll_freq_hz",
        "rms_gfl_pll_freq_hz",
        "max_abs_gfm_freq_hz",
        "max_gfl_current_pu",
        "max_gfm_current_pu",
        "gfl_limiter_fraction",
        "gfm_limiter_fraction",
    ]

    for m in metrics:
        result[f"delta_{m}"] = np.nan

    for sid in result[
        "scenario_id"
    ].unique():
        base = result[
            (result["scenario_id"] == sid)
            & (
                result["window_mode"]
                == "normal_counterfactual"
            )
        ]

        event = result[
            (result["scenario_id"] == sid)
            & (
                result["window_mode"]
                == "event"
            )
        ]

        if base.empty or event.empty:
            continue

        bi = base.index[0]
        ei = event.index[0]

        for m in metrics:
            b = float(result.at[bi, m])
            e = float(result.at[ei, m])

            result.at[
                bi,
                f"delta_{m}"
            ] = 0.0

            result.at[
                ei,
                f"delta_{m}"
            ] = e - b

    return result


def make_metadata_row(
    scenario_id,
    event_type,
    context,
    seed,
    p,
):
    row = {
        "scenario_id": scenario_id,
        "scenario_seed": seed,
        "event_type": event_type,
        "event_family": EVENT_FAMILY[
            event_type
        ],

        "context_timestamp": context[
            "timestamp"
        ],
        "context_pv_p_mw": float(
            context["pv_p_mw"]
        ),
        "context_load_total_p_mw": float(
            context["load_total_p_mw"]
        ),
        "context_bess_soc": float(
            context["bess_soc"]
        ),
    }

    row.update(p)
    return row


def plot_population_fingerprint(summary):
    d = summary[
        summary["window_mode"] == "event"
    ].copy()

    fig, ax = plt.subplots(
        figsize=(10.5, 7)
    )

    for event_type in EVENT_TYPES:
        x = d[
            d["event_type"] == event_type
        ]

        if x.empty:
            continue

        ax.scatter(
            x["max_abs_gfl_pll_freq_hz"],
            x["max_abs_gfm_freq_hz"],
            s=48,
            alpha=0.75,
            label=event_type,
        )

    ax.set_xlabel(
        "Maximum |GFL PLL frequency deviation| (Hz)"
    )
    ax.set_ylabel(
        "Maximum |GFM frequency deviation| (Hz)"
    )
    ax.set_title(
        "Randomized Cyber-Physical Event Population — Inverter Fingerprint Space"
    )
    ax.legend(
        frameon=False,
        fontsize=8,
        ncol=2,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(
        FIGURES
        / "22_randomized_event_population_fingerprint.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_context_coverage(metadata):
    fig, ax = plt.subplots(
        figsize=(9.5, 6.5)
    )

    sc = ax.scatter(
        1000*metadata["context_load_total_p_mw"],
        1000*metadata["context_pv_p_mw"],
        c=100*metadata["context_bess_soc"],
        s=55,
        alpha=0.80,
    )

    ax.set_xlabel("Context total load (kW)")
    ax.set_ylabel("Context PV power (kW)")
    ax.set_title(
        "Randomized Benchmark Operating-Point Coverage"
    )

    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("BESS SOC (%)")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(
        FIGURES
        / "23_randomized_operating_point_coverage.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_matched_delta_map(summary):
    d = summary[
        summary["window_mode"] == "event"
    ].copy()

    fig, ax = plt.subplots(
        figsize=(10.5, 7)
    )

    for event_type in EVENT_TYPES:
        x = d[
            d["event_type"] == event_type
        ]

        if x.empty:
            continue

        ax.scatter(
            x["delta_min_bus_v_pu"],
            x["delta_max_abs_gfl_pll_freq_hz"],
            s=48,
            alpha=0.75,
            label=event_type,
        )

    ax.axvline(0.0, linewidth=0.8)
    ax.axhline(0.0, linewidth=0.8)

    ax.set_xlabel(
        "Matched change in minimum bus voltage (pu)"
    )
    ax.set_ylabel(
        "Matched change in max |PLL frequency deviation| (Hz)"
    )
    ax.set_title(
        "Counterfactual Event Effects Across the Randomized Population"
    )

    ax.legend(
        frameon=False,
        fontsize=8,
        ncol=2,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(
        FIGURES
        / "24_randomized_matched_effect_map.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--n-per-class",
        type=int,
        default=6,
        help=(
            "Number of randomized scenarios "
            "per event type."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=20260811,
    )

    parser.add_argument(
        "--save-raw",
        action="store_true",
        help=(
            "Save full 1-ms transient rows. "
            "Summary/metadata are always saved."
        ),
    )

    args = parser.parse_args()

    slow_path = (
        DATA
        / "phase2e_7day_operating_dataset.csv"
    )

    if not slow_path.exists():
        raise FileNotFoundError(
            "Run Phase 2E research-fixed first: "
            + str(slow_path)
        )

    slow = pd.read_csv(
        slow_path,
        parse_dates=["timestamp"],
    )

    # Phase 2E event labels are only calendar labels now,
    # but use normal rows for maximally clean context sampling.
    contexts = slow[
        slow["event_label"] == "normal"
    ].reset_index(drop=True)

    if contexts.empty:
        raise RuntimeError(
            "No normal operating contexts available."
        )

    master_rng = np.random.default_rng(
        args.seed
    )

    metadata_rows = []
    summary_rows = []
    raw_frames = []

    scenario_id = 0

    total = (
        args.n_per_class
        * len(EVENT_TYPES)
    )

    print(
        "\n=== PHASE 2G: RANDOMIZED MATCHED BENCHMARK ==="
    )
    print(
        f"Event types             : {len(EVENT_TYPES)}"
    )
    print(
        f"Scenarios per event type: {args.n_per_class}"
    )
    print(
        f"Total abnormal scenarios: {total}"
    )
    print(
        f"Matched windows total   : {2*total}"
    )
    print(
        f"Fast integration        : {CONTROL_DT*1000:.1f} ms"
    )
    print(
        f"AC network refresh      : {NETWORK_DT*1000:.0f} ms"
    )
    print(
        f"Save raw 1-ms data      : {args.save_raw}"
    )

    for event_type in EVENT_TYPES:
        print(
            f"\n--- {event_type} ---"
        )

        for j in range(
            args.n_per_class
        ):
            seed = int(
                master_rng.integers(
                    0,
                    2**31 - 1
                )
            )

            rng = np.random.default_rng(seed)

            context_idx = int(
                rng.integers(
                    0,
                    len(contexts)
                )
            )
            context = contexts.iloc[
                context_idx
            ]

            p = sample_scenario_params(
                rng,
                event_type
            )

            n_steps = int(
                np.floor(
                    p["sim_end_s"]
                    / CONTROL_DT
                )
            ) + 1

            # Same stochastic realization in event/counterfactual.
            noise_v = rng.normal(
                0.0,
                p["v_noise_sigma_pu"],
                n_steps,
            )

            noise_theta_rad = np.deg2rad(
                rng.normal(
                    0.0,
                    p["angle_noise_sigma_deg"],
                    n_steps,
                )
            )

            metadata_rows.append(
                make_metadata_row(
                    scenario_id,
                    event_type,
                    context,
                    seed,
                    p,
                )
            )

            for mode in [
                "normal_counterfactual",
                "event",
            ]:
                df = run_window(
                    scenario_id,
                    event_type,
                    context,
                    p,
                    mode,
                    noise_v,
                    noise_theta_rad,
                )

                summary_rows.append(
                    summarize(df, p)
                )

                if args.save_raw:
                    raw_frames.append(df)

            print(
                f"  {j+1:02d}/{args.n_per_class} "
                f"scenario_id={scenario_id} | "
                f"PV={1000*context['pv_p_mw']:.1f} kW | "
                f"Load={1000*context['load_total_p_mw']:.1f} kW | "
                f"SOC={100*context['bess_soc']:.1f}% | "
                f"dur={p['event_duration_s']:.3f}s"
            )

            scenario_id += 1

    metadata = pd.DataFrame(
        metadata_rows
    )
    summary = pd.DataFrame(
        summary_rows
    )

    summary = add_matched_deltas(
        summary
    )

    metadata.to_csv(
        DATA
        / "phase2g_randomized_scenario_metadata.csv",
        index=False,
    )

    summary.to_csv(
        RESULTS
        / "phase2g_randomized_matched_summary.csv",
        index=False,
    )

    if args.save_raw and raw_frames:
        raw = pd.concat(
            raw_frames,
            ignore_index=True,
        )

        raw.to_csv(
            DATA
            / "phase2g_randomized_transient_timeseries.csv",
            index=False,
        )

    plot_population_fingerprint(summary)
    plot_context_coverage(metadata)
    plot_matched_delta_map(summary)

    event_summary = summary[
        summary["window_mode"] == "event"
    ].groupby(
        ["event_type", "event_family"],
        as_index=False,
    ).agg(
        n=("scenario_id", "count"),
        mean_context_soc=(
            "context_bess_soc",
            "mean",
        ),
        min_context_soc=(
            "context_bess_soc",
            "min",
        ),
        max_context_soc=(
            "context_bess_soc",
            "max",
        ),
        mean_min_bus_v=(
            "min_bus_v_pu",
            "mean",
        ),
        mean_max_pll_df=(
            "max_abs_gfl_pll_freq_hz",
            "mean",
        ),
        mean_max_gfm_df=(
            "max_abs_gfm_freq_hz",
            "mean",
        ),
        mean_max_gfl_current=(
            "max_gfl_current_pu",
            "mean",
        ),
        mean_max_gfm_current=(
            "max_gfm_current_pu",
            "mean",
        ),
        mean_delta_vmin=(
            "delta_min_bus_v_pu",
            "mean",
        ),
    )

    event_summary.to_csv(
        RESULTS
        / "phase2g_population_summary_by_event.csv",
        index=False,
    )

    print(
        "\n=== POPULATION SUMMARY ==="
    )
    print(
        event_summary.to_string(
            index=False
        )
    )

    print(
        "\nOperating-point coverage:"
    )
    print(
        f"  PV   : "
        f"{1000*metadata['context_pv_p_mw'].min():.1f} .. "
        f"{1000*metadata['context_pv_p_mw'].max():.1f} kW"
    )
    print(
        f"  Load : "
        f"{1000*metadata['context_load_total_p_mw'].min():.1f} .. "
        f"{1000*metadata['context_load_total_p_mw'].max():.1f} kW"
    )
    print(
        f"  SOC  : "
        f"{100*metadata['context_bess_soc'].min():.1f} .. "
        f"{100*metadata['context_bess_soc'].max():.1f}%"
    )

    print("\nSaved:")
    print(
        "  data/phase2g_randomized_scenario_metadata.csv"
    )
    print(
        "  results/phase2g_randomized_matched_summary.csv"
    )
    print(
        "  results/phase2g_population_summary_by_event.csv"
    )
    if args.save_raw:
        print(
            "  data/phase2g_randomized_transient_timeseries.csv"
        )
    print(
        "  figures/22_randomized_event_population_fingerprint.png"
    )
    print(
        "  figures/23_randomized_operating_point_coverage.png"
    )
    print(
        "  figures/24_randomized_matched_effect_map.png"
    )

    print(
        "\nNEXT:\n"
        "If the pilot population is numerically healthy and operating-point "
        "coverage is sufficiently diverse, rerun the SAME generator at larger "
        "scale with raw transient storage enabled. Only then should temporal, "
        "WLS-residual and graph/topology feature engineering begin."
    )


if __name__ == "__main__":
    main()
