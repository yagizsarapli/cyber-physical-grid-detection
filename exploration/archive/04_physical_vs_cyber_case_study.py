from pathlib import Path
import importlib.util
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)


def load_module(filename, module_name):
    spec = importlib.util.spec_from_file_location(module_name, ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


inv = load_module("02_inverter_models.py", "inverter_models")

DT = 1e-3
SIM_END = 4.0

EVENT_START = 1.0
EVENT_END = 1.20

V_NOMINAL = 1.0
V_SAG = 0.55
ANGLE_JUMP_DEG = 8.0
ANGLE_JUMP_RAD = np.deg2rad(ANGLE_JUMP_DEG)

PV_P_REF_PU = 0.35
BESS_P_REF_PU = 0.15

CYBER_SCORE_THRESHOLD = 1.0


def wrap_angle(angle_rad):
    return (angle_rad + np.pi) % (2*np.pi) - np.pi


def event_active(t):
    return EVENT_START <= t < EVENT_END


def scenario_signals(t, case):
    if case == "physical":
        if event_active(t):
            v_true = V_SAG
            th_true = ANGLE_JUMP_RAD
        else:
            v_true = V_NOMINAL
            th_true = 0.0

        v_gfl_meas = v_true
        th_gfl_meas = th_true

        # Independent/reference channel sees the same real physical event.
        v_ref = v_true
        th_ref = th_true

    elif case == "cyber":
        # Physical grid remains healthy.
        v_true = V_NOMINAL
        th_true = 0.0

        # Only the GFL measurement channel is spoofed.
        if event_active(t):
            v_gfl_meas = V_SAG
            th_gfl_meas = ANGLE_JUMP_RAD
        else:
            v_gfl_meas = V_NOMINAL
            th_gfl_meas = 0.0

        v_ref = v_true
        th_ref = th_true

    else:
        raise ValueError(case)

    return v_true, th_true, v_gfl_meas, th_gfl_meas, v_ref, th_ref


def run_case(case):
    gfl = inv.GFLInverter(
        kp_pll=18.0,
        ki_pll=250.0,
        i_max_pu=1.20,
        q_voltage_gain=2.0,
    )

    gfm = inv.GFMInverter(
        m_inertia=0.25,
        d_damping=0.80,
        e_internal_pu=1.02,
        x_virtual_pu=0.25,
        i_max_pu=1.20,
    )

    gfl.reset(theta0=0.0)

    arg = BESS_P_REF_PU * gfm.x_virtual_pu / gfm.e_internal_pu
    theta_gfm_0 = np.arcsin(np.clip(arg, -0.95, 0.95))
    gfm.reset(theta0=theta_gfm_0)

    rows = []

    for t in np.arange(0.0, SIM_END + DT/2, DT):
        (
            v_true,
            th_true,
            v_gfl_meas,
            th_gfl_meas,
            v_ref,
            th_ref,
        ) = scenario_signals(t, case)

        gfl_out = gfl.step(
            v_meas_pu=v_gfl_meas,
            theta_meas_rad=th_gfl_meas,
            p_ref_pu=PV_P_REF_PU,
            q_ref_pu=0.0,
            dt=DT,
        )

        gfm_out = gfm.step(
            v_pcc_pu=v_true,
            theta_pcc_rad=th_true,
            p_ref_pu=BESS_P_REF_PU,
            dt=DT,
        )

        dv_norm = abs(v_gfl_meas - v_ref) / 0.05
        dtheta_deg = abs(np.rad2deg(wrap_angle(th_gfl_meas - th_ref)))
        dtheta_norm = dtheta_deg / 5.0

        consistency_score = dv_norm + dtheta_norm
        cyber_flag = consistency_score >= CYBER_SCORE_THRESHOLD

        rows.append({
            "case": case,
            "time_s": t,
            "event_active": event_active(t),

            "v_true_pu": v_true,
            "theta_true_deg": np.rad2deg(th_true),
            "v_gfl_measured_pu": v_gfl_meas,
            "theta_gfl_measured_deg": np.rad2deg(th_gfl_meas),
            "v_reference_pu": v_ref,
            "theta_reference_deg": np.rad2deg(th_ref),

            "gfl_pll_angle_deg": gfl_out["theta_pll_deg"],
            "gfl_pll_freq_dev_hz": gfl_out["pll_freq_dev_hz"],
            "gfl_id_pu": gfl_out["id_pu"],
            "gfl_iq_pu": gfl_out["iq_pu"],
            "gfl_current_pu": gfl_out["i_mag_pu"],
            "gfl_limiter_active": gfl_out["limiter_active"],

            "gfm_internal_angle_deg": gfm_out["theta_internal_deg"],
            "gfm_freq_dev_hz": gfm_out["gfm_freq_dev_hz"],
            "gfm_p_pu": gfm_out["p_out_pu"],
            "gfm_q_pu": gfm_out["q_out_pu"],
            "gfm_current_pu": gfm_out["i_mag_pu"],
            "gfm_limiter_active": gfm_out["limiter_active"],

            "consistency_score": consistency_score,
            "cyber_flag": cyber_flag,
        })

    return pd.DataFrame(rows)


def first_detection_time(df):
    hit = df.loc[df["cyber_flag"], "time_s"]
    return None if hit.empty else float(hit.iloc[0])


def summarize(df, case):
    event = df[df["event_active"]]

    print(f"\n=== {case.upper()} CASE ===")
    print(f"max |GFL PLL freq dev| : {event['gfl_pll_freq_dev_hz'].abs().max():.4f} Hz")
    print(f"max |GFL iq|           : {event['gfl_iq_pu'].abs().max():.4f} pu")
    print(f"max GFL current        : {event['gfl_current_pu'].max():.4f} pu")
    print(f"GFL limiter active     : {bool(event['gfl_limiter_active'].any())}")

    print(f"max |GFM freq dev|     : {event['gfm_freq_dev_hz'].abs().max():.4f} Hz")
    print(f"max GFM current        : {event['gfm_current_pu'].max():.4f} pu")
    print(f"GFM limiter active     : {bool(event['gfm_limiter_active'].any())}")

    print(f"max consistency score  : {event['consistency_score'].max():.3f}")

    tdet = first_detection_time(df)
    print("first cyber flag       : " + ("none" if tdet is None else f"{tdet:.3f} s"))


def save_plot(x, series, ylabel, title, filename, threshold=None):
    fig, ax = plt.subplots(figsize=(10, 5.8))
    for y, label, linestyle in series:
        ax.plot(x, y, linewidth=2.2, linestyle=linestyle, label=label)

    if threshold is not None:
        ax.axhline(threshold, linestyle="--", linewidth=1.3, label="Detection threshold")

    ax.axvspan(EVENT_START, EVENT_END, alpha=0.10)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES / filename, dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    physical = run_case("physical")
    cyber = run_case("cyber")

    combined = pd.concat([physical, cyber], ignore_index=True)
    combined.to_csv(RESULTS / "physical_vs_cyber_case_study.csv", index=False)

    summarize(physical, "physical")
    summarize(cyber, "cyber")

    p = physical
    c = cyber

    save_plot(
        p["time_s"],
        [
            (p["v_true_pu"], "Physical case — true PCC voltage", "-"),
            (c["v_true_pu"], "Cyber case — true PCC voltage", "-"),
            (c["v_gfl_measured_pu"], "Cyber case — spoofed GFL measurement", "--"),
        ],
        "Voltage (pu)",
        "Physical Voltage Sag vs Spoofed GFL Voltage Measurement",
        "01_physical_vs_cyber_voltage.png",
    )

    save_plot(
        p["time_s"],
        [
            (p["gfl_pll_angle_deg"], "Physical voltage sag", "-"),
            (c["gfl_pll_angle_deg"], "Spoofed measurement attack", "--"),
        ],
        "GFL PLL angle (deg)",
        "GFL Internal PLL Response",
        "02_gfl_pll_physical_vs_cyber.png",
    )

    save_plot(
        p["time_s"],
        [
            (p["gfm_freq_dev_hz"], "Physical voltage sag", "-"),
            (c["gfm_freq_dev_hz"], "Spoofed GFL measurement attack", "--"),
        ],
        "GFM frequency deviation (Hz)",
        "GFM Physical Response Separates Real Events from Sensor Attacks",
        "03_gfm_frequency_physical_vs_cyber.png",
    )

    save_plot(
        p["time_s"],
        [
            (p["consistency_score"], "Physical voltage sag", "-"),
            (c["consistency_score"], "Spoofed measurement attack", "--"),
        ],
        "Cross-channel consistency score",
        "Cyber-Physical Consistency Residual",
        "04_cyberphysical_consistency_score.png",
        threshold=CYBER_SCORE_THRESHOLD,
    )

    print("\nSaved:")
    print("  results/physical_vs_cyber_case_study.csv")
    print("  figures/01_physical_vs_cyber_voltage.png")
    print("  figures/02_gfl_pll_physical_vs_cyber.png")
    print("  figures/03_gfm_frequency_physical_vs_cyber.png")
    print("  figures/04_cyberphysical_consistency_score.png")

    print(
        "\nIMPORTANT: the consistency score currently uses an independent "
        "reference channel. In the next phase this will be replaced by "
        "state-estimation/network residuals."
    )
