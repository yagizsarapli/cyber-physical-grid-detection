from pathlib import Path
import importlib.util
import argparse
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


base = load_module(
    "10_randomized_matched_benchmark.py",
    "phase2g_base"
)

EVENT_TYPES = base.EVENT_TYPES
EVENT_FAMILY = base.EVENT_FAMILY


# ============================================================
# PHASE 2G-V2 — DATASET-GRADE RANDOMIZED MATCHED BENCHMARK
# ============================================================
#
# Improvements over the pilot:
#   1) Event-specific operating-context eligibility.
#   2) No cloud/GFL event is sampled at a meaningless zero-PV context.
#   3) GFL current-limit manipulation is scaled to the actual operating current.
#   4) Matched normal/event windows remain paired by scenario_id.
#   5) Group-safe train/validation/test split by scenario_id.
#   6) Scenario quality gates expose degenerate/no-effect samples.
#   7) Simulation remains at 1 ms; optional raw storage is decimated
#      to a user-selected interval (default 5 ms) and gzip-compressed.
# ============================================================

MIN_PV_CLOUD_MW = 0.012
MIN_PV_GFL_EVENT_MW = 0.010
MIN_PV_GFL_LIMIT_MW = 0.015


def eligible_contexts(contexts, event_type):
    c = contexts.copy()

    if event_type == "physical_cloud_transient":
        c = c[
            c["pv_p_mw"] >= MIN_PV_CLOUD_MW
        ]

    elif event_type in (
        "cyber_fake_voltage_measurement",
        "cyber_phase_drift",
    ):
        c = c[
            c["pv_p_mw"] >= MIN_PV_GFL_EVENT_MW
        ]

    elif event_type == "cyber_gfl_current_limit":
        c = c[
            c["pv_p_mw"] >= MIN_PV_GFL_LIMIT_MW
        ]

    elif event_type == "hybrid_fault_masking":
        # The scientific purpose of this class is to observe a physically
        # active fault being hidden from an operating GFL inverter.
        c = c[
            c["pv_p_mw"] >= MIN_PV_GFL_EVENT_MW
        ]

    if c.empty:
        raise RuntimeError(
            f"No eligible operating contexts for {event_type}"
        )

    return c.reset_index(drop=True)


def sample_context_and_params(
    master_rng,
    contexts,
    event_type,
):
    seed = int(
        master_rng.integers(
            0,
            2**31 - 1
        )
    )
    rng = np.random.default_rng(seed)

    eligible = eligible_contexts(
        contexts,
        event_type
    )

    context_idx = int(
        rng.integers(
            0,
            len(eligible)
        )
    )
    context = eligible.iloc[context_idx]

    p = base.sample_scenario_params(
        rng,
        event_type
    )

    # Make current-limit manipulation operating-point aware.
    # Approximate pre-event current is dominated by active current:
    # I ~= P / V on the per-unit base.
    if event_type == "cyber_gfl_current_limit":
        v_context = float(
            context.get(
                "bus_1_vm_pu",
                0.98
            )
        )
        p_pu = (
            float(context["pv_p_mw"])
            / base.S_BASE_MVA
        )
        expected_i_pu = (
            p_pu
            / max(v_context, 0.90)
        )

        fraction = float(
            rng.uniform(0.35, 0.72)
        )

        p["gfl_attack_imax_pu"] = float(
            np.clip(
                fraction * expected_i_pu,
                0.05,
                0.60,
            )
        )

        p[
            "estimated_pre_event_gfl_current_pu"
        ] = expected_i_pu

    else:
        p[
            "estimated_pre_event_gfl_current_pu"
        ] = np.nan

    return seed, rng, context, p


def assign_group_splits(
    metadata,
    seed,
):
    """
    Split by scenario_id, stratified by event type.

    The matched normal/event pair always receives the same split,
    preventing counterfactual leakage across train/test.
    """
    split_map = {}
    rng = np.random.default_rng(
        seed + 991
    )

    for event_type in EVENT_TYPES:
        ids = (
            metadata.loc[
                metadata["event_type"]
                == event_type,
                "scenario_id"
            ]
            .astype(int)
            .to_numpy()
        )

        ids = rng.permutation(ids)
        n = len(ids)

        if n < 3:
            for sid in ids:
                split_map[int(sid)] = "train"
            continue

        n_train = max(
            1,
            int(np.floor(0.70 * n))
        )
        n_val = max(
            1,
            int(np.floor(0.15 * n))
        )

        if n_train + n_val >= n:
            n_train = max(1, n - 2)
            n_val = 1

        train_ids = ids[:n_train]
        val_ids = ids[
            n_train:n_train+n_val
        ]
        test_ids = ids[
            n_train+n_val:
        ]

        for sid in train_ids:
            split_map[int(sid)] = "train"
        for sid in val_ids:
            split_map[int(sid)] = "validation"
        for sid in test_ids:
            split_map[int(sid)] = "test"

    return split_map


def build_quality_audit(
    summary,
    metadata,
):
    """
    Defensive data-quality checks.

    These gates do not claim attack-detection performance.
    They only verify that a generated event is non-degenerate and
    physically/algorithmically meaningful enough to retain.
    """
    event_rows = summary[
        summary["window_mode"] == "event"
    ].copy()

    meta = metadata.set_index(
        "scenario_id"
    )

    rows = []

    for _, r in event_rows.iterrows():
        sid = int(r["scenario_id"])
        et = str(r["event_type"])
        m = meta.loc[sid]

        passed = True
        reason = "ok"

        if et == "physical_cloud_transient":
            passed = (
                float(r["context_pv_p_mw"])
                >= MIN_PV_CLOUD_MW
            )
            if not passed:
                reason = "insufficient_PV_for_cloud_event"

        elif et == "physical_load_spike":
            passed = (
                float(
                    r["delta_min_bus_v_pu"]
                )
                < -0.0005
            )
            if not passed:
                reason = "load_spike_has_negligible_grid_effect"

        elif et == "physical_voltage_sag":
            passed = (
                float(
                    r["delta_min_bus_v_pu"]
                )
                < -0.05
            )
            if not passed:
                reason = "voltage_sag_effect_too_small"

        elif et == "cyber_fake_voltage_measurement":
            passed = (
                float(
                    r[
                        "max_abs_sensor_v_residual_pu"
                    ]
                )
                > 0.04
                or float(
                    r[
                        "max_abs_sensor_angle_residual_deg"
                    ]
                )
                > 1.0
            )
            if not passed:
                reason = "fake_measurement_not_observable"

        elif et == "cyber_phase_drift":
            passed = (
                float(
                    r[
                        "max_abs_sensor_angle_residual_deg"
                    ]
                )
                > 1.0
            )
            if not passed:
                reason = "phase_drift_too_small"

        elif et == "cyber_gfl_current_limit":
            passed = (
                float(
                    r["gfl_limiter_fraction"]
                )
                > 0.05
            )
            if not passed:
                reason = "current_limit_not_binding"

        elif et == "cyber_gfm_setpoint_manipulation":
            passed = (
                abs(
                    float(
                        r[
                            "delta_max_abs_gfm_freq_hz"
                        ]
                    )
                )
                > 0.002
                or abs(
                    float(
                        r[
                            "delta_max_gfm_current_pu"
                        ]
                    )
                )
                > 0.01
            )
            if not passed:
                reason = "gfm_setpoint_effect_too_small"

        elif et == "hybrid_fault_masking":
            passed = (
                float(r["context_pv_p_mw"])
                >= MIN_PV_GFL_EVENT_MW
                and float(
                    r["delta_min_bus_v_pu"]
                )
                < -0.05
                and float(
                    r[
                        "max_abs_sensor_v_residual_pu"
                    ]
                )
                > 0.05
            )
            if not passed:
                reason = "hybrid_masking_not_distinct_or_PV_inactive"

        rows.append({
            "scenario_id": sid,
            "event_type": et,
            "event_family": EVENT_FAMILY[
                et
            ],
            "quality_pass": bool(passed),
            "quality_reason": reason,
            "context_pv_p_mw": float(
                r["context_pv_p_mw"]
            ),
            "context_load_total_p_mw": float(
                r["context_load_total_p_mw"]
            ),
            "context_bess_soc": float(
                r["context_bess_soc"]
            ),
            "gfl_attack_imax_pu": float(
                m["gfl_attack_imax_pu"]
            ),
        })

    return pd.DataFrame(rows)


def plot_coverage(metadata):
    fig, ax = plt.subplots(
        figsize=(9.5, 6.5)
    )

    sc = ax.scatter(
        1000
        * metadata[
            "context_load_total_p_mw"
        ],
        1000
        * metadata[
            "context_pv_p_mw"
        ],
        c=100
        * metadata[
            "context_bess_soc"
        ],
        s=50,
        alpha=0.78,
    )

    ax.set_xlabel(
        "Context total load (kW)"
    )
    ax.set_ylabel(
        "Context PV power (kW)"
    )
    ax.set_title(
        "Dataset-Grade Randomized Operating-Point Coverage"
    )

    cbar = fig.colorbar(
        sc,
        ax=ax
    )
    cbar.set_label(
        "BESS SOC (%)"
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(
        FIGURES
        / "25_v2_operating_point_coverage.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_quality(audit):
    d = (
        audit.groupby(
            "event_type",
            as_index=False
        )["quality_pass"]
        .mean()
    )

    fig, ax = plt.subplots(
        figsize=(11, 6)
    )

    x = np.arange(len(d))
    values = (
        100
        * d["quality_pass"].to_numpy()
    )

    ax.bar(
        x,
        values
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        d["event_type"],
        rotation=25,
        ha="right",
    )
    ax.set_ylabel(
        "Scenario quality-pass rate (%)"
    )
    ax.set_ylim(
        0,
        105
    )
    ax.set_title(
        "Generated-Scenario Non-Degeneracy Audit"
    )

    for i, value in enumerate(values):
        ax.text(
            i,
            min(value + 1.5, 102),
            f"{value:.0f}%",
            ha="center",
            fontsize=9,
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(
        FIGURES
        / "26_v2_scenario_quality_audit.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_fingerprint(summary):
    d = summary[
        summary["window_mode"] == "event"
    ]

    fig, ax = plt.subplots(
        figsize=(10.5, 7)
    )

    for event_type in EVENT_TYPES:
        x = d[
            d["event_type"]
            == event_type
        ]

        if x.empty:
            continue

        ax.scatter(
            x[
                "max_abs_gfl_pll_freq_hz"
            ],
            x[
                "max_abs_gfm_freq_hz"
            ],
            s=46,
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
        "Dataset-Grade Randomized Inverter Fingerprint Population"
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
        / "27_v2_event_fingerprint_population.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--n-per-class",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=20260812,
    )

    parser.add_argument(
        "--save-raw",
        action="store_true",
    )

    parser.add_argument(
        "--raw-step-ms",
        type=float,
        default=5.0,
        help=(
            "Raw output sampling interval. "
            "Simulation itself always remains at 1 ms."
        ),
    )

    args = parser.parse_args()

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

    contexts = (
        slow[
            slow["event_label"]
            == "normal"
        ]
        .reset_index(drop=True)
    )

    if contexts.empty:
        raise RuntimeError(
            "No clean operating contexts available."
        )

    master_rng = np.random.default_rng(
        args.seed
    )

    metadata_rows = []
    summary_rows = []
    raw_frames = []

    raw_stride = max(
        1,
        int(
            round(
                args.raw_step_ms
                / (
                    base.CONTROL_DT
                    * 1000.0
                )
            )
        )
    )

    total = (
        args.n_per_class
        * len(EVENT_TYPES)
    )

    print(
        "\n=== PHASE 2G-V2: DATASET-GRADE RANDOMIZED BENCHMARK ==="
    )
    print(
        f"Event types              : {len(EVENT_TYPES)}"
    )
    print(
        f"Scenarios per event type : {args.n_per_class}"
    )
    print(
        f"Abnormal scenarios       : {total}"
    )
    print(
        f"Matched windows          : {2*total}"
    )
    print(
        f"Simulation step          : {base.CONTROL_DT*1000:.1f} ms"
    )
    print(
        f"AC-network refresh       : {base.NETWORK_DT*1000:.0f} ms"
    )
    print(
        f"Save raw                 : {args.save_raw}"
    )
    if args.save_raw:
        print(
            f"Stored raw interval      : "
            f"{raw_stride*base.CONTROL_DT*1000:.1f} ms"
        )

    scenario_id = 0

    for event_type in EVENT_TYPES:
        print(
            f"\n--- {event_type} ---"
        )

        for j in range(
            args.n_per_class
        ):
            (
                seed,
                rng,
                context,
                p,
            ) = sample_context_and_params(
                master_rng,
                contexts,
                event_type,
            )

            n_steps = int(
                np.floor(
                    p["sim_end_s"]
                    / base.CONTROL_DT
                )
            ) + 1

            # Matched event/counterfactual use exactly the same noise.
            noise_v = rng.normal(
                0.0,
                p["v_noise_sigma_pu"],
                n_steps,
            )

            noise_theta_rad = np.deg2rad(
                rng.normal(
                    0.0,
                    p[
                        "angle_noise_sigma_deg"
                    ],
                    n_steps,
                )
            )

            metadata_rows.append(
                base.make_metadata_row(
                    scenario_id,
                    event_type,
                    context,
                    seed,
                    p,
                )
            )

            for mode in (
                "normal_counterfactual",
                "event",
            ):
                df = base.run_window(
                    scenario_id,
                    event_type,
                    context,
                    p,
                    mode,
                    noise_v,
                    noise_theta_rad,
                )

                summary_rows.append(
                    base.summarize(
                        df,
                        p
                    )
                )

                if args.save_raw:
                    stored = (
                        df.iloc[
                            ::raw_stride
                        ]
                        .copy()
                    )
                    raw_frames.append(
                        stored
                    )

            print(
                f"  {j+1:02d}/{args.n_per_class} "
                f"id={scenario_id} | "
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

    summary = base.add_matched_deltas(
        summary
    )

    split_map = assign_group_splits(
        metadata,
        args.seed
    )

    metadata["dataset_split"] = (
        metadata[
            "scenario_id"
        ].map(split_map)
    )

    summary["dataset_split"] = (
        summary[
            "scenario_id"
        ].map(split_map)
    )

    audit = build_quality_audit(
        summary,
        metadata
    )

    audit["dataset_split"] = (
        audit[
            "scenario_id"
        ].map(split_map)
    )

    metadata.to_csv(
        DATA
        / "phase2g_v2_scenario_metadata.csv",
        index=False,
    )

    summary.to_csv(
        RESULTS
        / "phase2g_v2_matched_summary.csv",
        index=False,
    )

    audit.to_csv(
        RESULTS
        / "phase2g_v2_quality_audit.csv",
        index=False,
    )

    if args.save_raw:
        raw = pd.concat(
            raw_frames,
            ignore_index=True,
        )

        raw["dataset_split"] = (
            raw[
                "scenario_id"
            ].map(split_map)
        )

        raw.to_csv(
            DATA
            / "phase2g_v2_transient_timeseries.csv.gz",
            index=False,
            compression="gzip",
        )

    plot_coverage(metadata)
    plot_quality(audit)
    plot_fingerprint(summary)

    quality_by_event = (
        audit.groupby(
            "event_type",
            as_index=False
        )
        .agg(
            n=(
                "scenario_id",
                "count",
            ),
            quality_pass_rate=(
                "quality_pass",
                "mean",
            ),
            min_pv_mw=(
                "context_pv_p_mw",
                "min",
            ),
            max_pv_mw=(
                "context_pv_p_mw",
                "max",
            ),
            min_soc=(
                "context_bess_soc",
                "min",
            ),
            max_soc=(
                "context_bess_soc",
                "max",
            ),
        )
    )

    split_counts = (
        metadata.groupby(
            [
                "event_type",
                "dataset_split"
            ],
            as_index=False
        )["scenario_id"]
        .count()
        .rename(
            columns={
                "scenario_id": "n"
            }
        )
    )

    print(
        "\n=== QUALITY AUDIT ==="
    )
    q = quality_by_event.copy()
    q[
        "quality_pass_rate"
    ] *= 100.0
    q[
        "min_pv_kw"
    ] = 1000*q["min_pv_mw"]
    q[
        "max_pv_kw"
    ] = 1000*q["max_pv_mw"]
    q[
        "min_soc_pct"
    ] = 100*q["min_soc"]
    q[
        "max_soc_pct"
    ] = 100*q["max_soc"]

    print(
        q[
            [
                "event_type",
                "n",
                "quality_pass_rate",
                "min_pv_kw",
                "max_pv_kw",
                "min_soc_pct",
                "max_soc_pct",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\n=== GROUP-SAFE DATASET SPLIT ==="
    )
    print(
        split_counts.to_string(
            index=False
        )
    )

    overall_pass = float(
        audit[
            "quality_pass"
        ].mean()
    )

    print(
        f"\nOverall scenario quality pass: "
        f"{100*overall_pass:.1f}%"
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

    print(
        "\nSaved:"
    )
    print(
        "  data/phase2g_v2_scenario_metadata.csv"
    )
    print(
        "  results/phase2g_v2_matched_summary.csv"
    )
    print(
        "  results/phase2g_v2_quality_audit.csv"
    )
    if args.save_raw:
        print(
            "  data/phase2g_v2_transient_timeseries.csv.gz"
        )
    print(
        "  figures/25_v2_operating_point_coverage.png"
    )
    print(
        "  figures/26_v2_scenario_quality_audit.png"
    )
    print(
        "  figures/27_v2_event_fingerprint_population.png"
    )

    print(
        "\nFEATURE POLICY:\n"
        "BESS SOC is retained as scenario metadata only in this dataset version. "
        "Do not use SOC as a detector input until the fast model includes "
        "SOC/DC-side dependent BESS constraints, otherwise it may become a "
        "non-causal shortcut feature."
    )

    print(
        "\nDECISION RULE:\n"
        "If every event type has a high non-degeneracy pass rate "
        "(ideally 100% in this pilot) and context coverage remains broad, "
        "the generator is ready for the full dataset run. "
        "The next full run should use --save-raw with 5-ms stored data; "
        "the simulator itself will still integrate inverter dynamics at 1 ms."
    )


if __name__ == "__main__":
    main()
