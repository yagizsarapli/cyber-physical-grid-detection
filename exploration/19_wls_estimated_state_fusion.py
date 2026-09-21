from pathlib import Path
import importlib.util
import copy
import itertools
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandapower as pp

from pandapower.estimation import estimate
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

DATA.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)

CHALLENGE_RAW_PATH = (
    DATA
    / "phase2m_ambiguous_challenge_timeseries.csv.gz"
)
CHALLENGE_META_PATH = (
    DATA
    / "phase2m_ambiguous_challenge_metadata.csv"
)
SLOW_CONTEXT_PATH = (
    DATA
    / "phase2e_7day_operating_dataset.csv"
)
PHASE2N_FEATURE_PATH = (
    DATA
    / "phase2n_fusion_feature_dataset.csv"
)

RANDOM_STATE = 20260812
S_BASE_MVA = 0.10
PF_LOAD = 0.95

SNAPSHOT_FRACTIONS = [
    0.25,
    0.50,
    0.75,
]

# Independent synthetic SCADA uncertainties.
STD_V_PU = 0.0050
STD_P_MW = 0.0010
STD_Q_MVAR = 0.0008
STD_LINE_P_MW = 0.0012
STD_LINE_Q_MVAR = 0.0010

POSITIVE_CLASS = (
    "cyber_mild_gfl_current_limit"
)


def load_module(
    filename,
    module_name,
):
    spec = (
        importlib.util
        .spec_from_file_location(
            module_name,
            ROOT / filename,
        )
    )
    module = (
        importlib.util
        .module_from_spec(
            spec
        )
    )
    spec.loader.exec_module(
        module
    )
    return module


topology = load_module(
    "01_microgrid_topology.py",
    "topology",
)


def reactive_from_pf(
    p_mw,
    pf=PF_LOAD,
):
    phi = np.arccos(pf)
    return float(
        p_mw
        * np.tan(phi)
    )


def context_lookup(
    slow,
):
    d = slow.copy()

    d[
        "timestamp"
    ] = pd.to_datetime(
        d[
            "timestamp"
        ]
    )

    return d.set_index(
        "timestamp"
    )


def nearest_raw_row(
    g,
    target_t,
):
    idx = (
        g[
            "time_s"
        ]
        .sub(
            target_t
        )
        .abs()
        .idxmin()
    )

    return g.loc[idx]


def build_reconstructed_truth_network(
    raw_row,
    meta_row,
    slow_row,
    mode,
):
    """
    Reconstruct a quasi-static AC snapshot from:

      - known slow operating context (loads),
      - observed inverter output trajectories,
      - physical source-voltage condition for the simulated truth generator.

    The detector will NEVER receive challenge labels or sag severity.
    They are used here only to synthesize the physical SCADA measurements,
    just as an EMT/physical plant would generate measurements in reality.
    """
    net = topology.build_microgrid()

    load_a = float(
        slow_row[
            "load_a_p_mw"
        ]
    )
    load_b = float(
        slow_row[
            "load_b_p_mw"
        ]
    )

    net.load.at[
        net.load.index[0],
        "p_mw",
    ] = load_a

    net.load.at[
        net.load.index[0],
        "q_mvar",
    ] = reactive_from_pf(
        load_a
    )

    net.load.at[
        net.load.index[1],
        "p_mw",
    ] = load_b

    net.load.at[
        net.load.index[1],
        "q_mvar",
    ] = reactive_from_pf(
        load_b
    )

    # Reconstruct GFL active/reactive output from independent trajectory channels.
    v_meas = float(
        raw_row[
            "v_gfl_measured_pu"
        ]
    )
    id_pu = float(
        raw_row[
            "gfl_id_pu"
        ]
    )
    iq_pu = float(
        raw_row[
            "gfl_iq_pu"
        ]
    )

    pv_p_mw = (
        v_meas
        * id_pu
        * S_BASE_MVA
    )

    # In the project GFL sign convention:
    # q_out_pu = -v * iq
    # pandapower sgen injection used in the loop is -q_out*Sbase,
    # hence +v*iq*Sbase here.
    pv_q_mvar = (
        v_meas
        * iq_pu
        * S_BASE_MVA
    )

    bess_p_mw = (
        float(
            raw_row[
                "gfm_p_pu"
            ]
        )
        * S_BASE_MVA
    )

    bess_q_mvar = (
        float(
            raw_row[
                "gfm_q_pu"
            ]
        )
        * S_BASE_MVA
    )

    pv_idx = net.gridra[
        "pv_sgen"
    ]
    bess_idx = net.gridra[
        "bess_sgen"
    ]

    net.sgen.at[
        pv_idx,
        "p_mw",
    ] = pv_p_mw

    net.sgen.at[
        pv_idx,
        "q_mvar",
    ] = pv_q_mvar

    net.sgen.at[
        bess_idx,
        "p_mw",
    ] = bess_p_mw

    net.sgen.at[
        bess_idx,
        "q_mvar",
    ] = bess_q_mvar

    # Physical deep-sag challenge:
    # challenge metadata is used only in the plant/truth reconstruction,
    # not as a classifier feature.
    ext = net.ext_grid.index[0]

    net.ext_grid.at[
        ext,
        "vm_pu",
    ] = 1.0

    net.ext_grid.at[
        ext,
        "va_degree",
    ] = 0.0

    if (
        mode == "event"
        and str(
            meta_row[
                "challenge_type"
            ]
        )
        == "physical_deep_sag_with_gfl_limiting"
    ):
        net.ext_grid.at[
            ext,
            "vm_pu",
        ] = float(
            meta_row[
                "physical_sag_vm_pu"
            ]
        )

    topology.solve_base_case(
        net
    )

    return net


def add_noisy_scada_measurements(
    estimator_net,
    truth_net,
    rng,
):
    """
    23 redundant measurements:
      - 5 bus voltage magnitudes
      - 5 bus P injections
      - 5 bus Q injections
      - 4 line P from-side flows
      - 4 line Q from-side flows
    """
    if len(
        estimator_net.measurement
    ):
        estimator_net.measurement.drop(
            estimator_net.measurement.index,
            inplace=True,
        )

    for bus in truth_net.bus.index:
        v = float(
            truth_net.res_bus.at[
                bus,
                "vm_pu",
            ]
        )
        p = float(
            truth_net.res_bus.at[
                bus,
                "p_mw",
            ]
        )
        q = float(
            truth_net.res_bus.at[
                bus,
                "q_mvar",
            ]
        )

        pp.create_measurement(
            estimator_net,
            meas_type="v",
            element_type="bus",
            value=float(
                v
                + rng.normal(
                    0.0,
                    STD_V_PU,
                )
            ),
            std_dev=STD_V_PU,
            element=int(
                bus
            ),
            name=f"V_bus_{bus}",
        )

        pp.create_measurement(
            estimator_net,
            meas_type="p",
            element_type="bus",
            value=float(
                p
                + rng.normal(
                    0.0,
                    STD_P_MW,
                )
            ),
            std_dev=STD_P_MW,
            element=int(
                bus
            ),
            name=f"P_bus_{bus}",
        )

        pp.create_measurement(
            estimator_net,
            meas_type="q",
            element_type="bus",
            value=float(
                q
                + rng.normal(
                    0.0,
                    STD_Q_MVAR,
                )
            ),
            std_dev=STD_Q_MVAR,
            element=int(
                bus
            ),
            name=f"Q_bus_{bus}",
        )

    for line in truth_net.line.index:
        p = float(
            truth_net.res_line.at[
                line,
                "p_from_mw",
            ]
        )
        q = float(
            truth_net.res_line.at[
                line,
                "q_from_mvar",
            ]
        )

        pp.create_measurement(
            estimator_net,
            meas_type="p",
            element_type="line",
            value=float(
                p
                + rng.normal(
                    0.0,
                    STD_LINE_P_MW,
                )
            ),
            std_dev=STD_LINE_P_MW,
            element=int(
                line
            ),
            side="from",
            name=f"P_line_{line}_from",
        )

        pp.create_measurement(
            estimator_net,
            meas_type="q",
            element_type="line",
            value=float(
                q
                + rng.normal(
                    0.0,
                    STD_LINE_Q_MVAR,
                )
            ),
            std_dev=STD_LINE_Q_MVAR,
            element=int(
                line
            ),
            side="from",
            name=f"Q_line_{line}_from",
        )


def measurement_residual_summary(
    net,
):
    rows = []

    for _, m in (
        net.measurement.iterrows()
    ):
        mt = str(
            m[
                "measurement_type"
            ]
        )
        et = str(
            m[
                "element_type"
            ]
        )
        element = int(
            m[
                "element"
            ]
        )
        measured = float(
            m[
                "value"
            ]
        )
        std = float(
            m[
                "std_dev"
            ]
        )

        estimated = np.nan

        if et == "bus":
            if mt == "v":
                estimated = float(
                    net.res_bus_est.at[
                        element,
                        "vm_pu",
                    ]
                )
            elif mt == "p":
                estimated = float(
                    net.res_bus_est.at[
                        element,
                        "p_mw",
                    ]
                )
            elif mt == "q":
                estimated = float(
                    net.res_bus_est.at[
                        element,
                        "q_mvar",
                    ]
                )

        elif et == "line":
            side = m[
                "side"
            ]

            if side not in (
                "from",
                "to",
            ):
                side = "from"

            if mt == "p":
                col = (
                    "p_from_mw"
                    if side == "from"
                    else "p_to_mw"
                )
                estimated = float(
                    net.res_line_est.at[
                        element,
                        col,
                    ]
                )

            elif mt == "q":
                col = (
                    "q_from_mvar"
                    if side == "from"
                    else "q_to_mvar"
                )
                estimated = float(
                    net.res_line_est.at[
                        element,
                        col,
                    ]
                )

        if np.isfinite(
            estimated
        ):
            normalized = abs(
                measured
                - estimated
            ) / max(
                std,
                1e-12,
            )

            rows.append(
                normalized
            )

    if not rows:
        return (
            np.nan,
            np.nan,
        )

    x = np.asarray(
        rows,
        dtype=float,
    )

    return (
        float(
            np.max(x)
        ),
        float(
            np.sqrt(
                np.mean(
                    x ** 2
                )
            )
        ),
    )


def estimate_snapshot(
    truth_net,
    rng,
):
    estimator_net = copy.deepcopy(
        truth_net
    )

    # Do not hand the estimator the physical sag magnitude through
    # the slack voltage setpoint. Let the redundant measurements
    # drive the estimated state; the slack/reference angle remains.
    ext = (
        estimator_net
        .ext_grid.index[0]
    )

    estimator_net.ext_grid.at[
        ext,
        "vm_pu",
    ] = 1.0

    estimator_net.ext_grid.at[
        ext,
        "va_degree",
    ] = 0.0

    add_noisy_scada_measurements(
        estimator_net,
        truth_net,
        rng,
    )

    try:
        success = bool(
            estimate(
                estimator_net,
                algorithm="wls",
                init="flat",
                tolerance=1e-7,
                maximum_iterations=30,
                calculate_voltage_angles=True,
            )
        )
    except Exception:
        success = False

    if not success:
        return None

    est_v = (
        estimator_net
        .res_bus_est[
            "vm_pu"
        ]
        .to_numpy(
            dtype=float
        )
    )

    true_v = (
        truth_net
        .res_bus[
            "vm_pu"
        ]
        .to_numpy(
            dtype=float
        )
    )

    v_rmse = float(
        np.sqrt(
            np.mean(
                (
                    est_v
                    - true_v
                )
                ** 2
            )
        )
    )

    max_nr, rms_nr = (
        measurement_residual_summary(
            estimator_net
        )
    )

    out = {
        "wls_min_v_pu": float(
            np.min(
                est_v
            )
        ),
        "truth_min_v_pu": float(
            np.min(
                true_v
            )
        ),
        "wls_v_rmse_pu": v_rmse,
        "wls_max_norm_residual":
            max_nr,
        "wls_rms_norm_residual":
            rms_nr,
    }

    for bus in (
        estimator_net.bus.index
    ):
        out[
            f"bus_{int(bus)}_vm_est_pu"
        ] = float(
            estimator_net
            .res_bus_est.at[
                bus,
                "vm_pu",
            ]
        )

        out[
            f"bus_{int(bus)}_va_est_deg"
        ] = float(
            estimator_net
            .res_bus_est.at[
                bus,
                "va_degree",
            ]
        )

        out[
            f"bus_{int(bus)}_p_est_mw"
        ] = float(
            estimator_net
            .res_bus_est.at[
                bus,
                "p_mw",
            ]
        )

        out[
            f"bus_{int(bus)}_q_est_mvar"
        ] = float(
            estimator_net
            .res_bus_est.at[
                bus,
                "q_mvar",
            ]
        )

    for line in (
        estimator_net.line.index
    ):
        out[
            f"line_{int(line)}_p_from_est_mw"
        ] = float(
            estimator_net
            .res_line_est.at[
                line,
                "p_from_mw",
            ]
        )

        out[
            f"line_{int(line)}_q_from_est_mvar"
        ] = float(
            estimator_net
            .res_line_est.at[
                line,
                "q_from_mvar",
            ]
        )

    return out


def build_wls_snapshots(
    challenge_raw,
    challenge_meta,
    slow,
):
    slow_idx = context_lookup(
        slow
    )

    meta_idx = (
        challenge_meta
        .set_index(
            "scenario_id"
        )
    )

    rows = []
    failed = 0

    grouped = (
        challenge_raw.groupby(
            [
                "scenario_id",
                "window_mode",
            ],
            sort=True,
        )
    )

    print(
        f"Running WLS on "
        f"{len(grouped) * len(SNAPSHOT_FRACTIONS)} "
        f"selected snapshots ..."
    )

    for (
        scenario_id,
        mode,
    ), g in grouped:
        scenario_id = int(
            scenario_id
        )

        m = meta_idx.loc[
            scenario_id
        ]

        timestamp = pd.to_datetime(
            m[
                "context_timestamp"
            ]
        )

        if timestamp not in (
            slow_idx.index
        ):
            raise KeyError(
                f"Context timestamp "
                f"{timestamp} not found "
                f"in Phase 2E dataset."
            )

        slow_row = (
            slow_idx.loc[
                timestamp
            ]
        )

        # If duplicate timestamp rows somehow exist,
        # use the first.
        if isinstance(
            slow_row,
            pd.DataFrame,
        ):
            slow_row = (
                slow_row.iloc[0]
            )

        t0 = float(
            m[
                "event_start_s"
            ]
        )
        t1 = float(
            m[
                "event_end_s"
            ]
        )

        duration = (
            t1 - t0
        )

        for j, frac in enumerate(
            SNAPSHOT_FRACTIONS
        ):
            target_t = (
                t0
                + frac
                * duration
            )

            raw_row = nearest_raw_row(
                g,
                target_t,
            )

            try:
                truth_net = (
                    build_reconstructed_truth_network(
                        raw_row,
                        m,
                        slow_row,
                        mode,
                    )
                )
            except Exception:
                failed += 1
                continue

            rng = (
                np.random.default_rng(
                    RANDOM_STATE
                    + 100000
                    * scenario_id
                    + 1000
                    * (
                        0
                        if mode
                        == "normal_counterfactual"
                        else 1
                    )
                    + j
                )
            )

            result = (
                estimate_snapshot(
                    truth_net,
                    rng,
                )
            )

            if result is None:
                failed += 1
                continue

            row = {
                "scenario_id":
                    scenario_id,
                "window_mode":
                    mode,
                "challenge_type":
                    str(
                        m[
                            "challenge_type"
                        ]
                    ),
                "snapshot_fraction":
                    frac,
                "snapshot_time_s":
                    float(
                        raw_row[
                            "time_s"
                        ]
                    ),
                **result,
            }

            rows.append(
                row
            )

    out = pd.DataFrame(
        rows
    )

    print(
        f"WLS successful snapshots: "
        f"{len(out)}"
    )
    print(
        f"WLS/reconstruction failures: "
        f"{failed}"
    )

    return out


def aggregate_wls_features(
    snapshots,
):
    agg = (
        snapshots.groupby(
            [
                "scenario_id",
                "window_mode",
                "challenge_type",
            ],
            as_index=False,
        )
        .agg(
            wls_min_v_pu=(
                "wls_min_v_pu",
                "min",
            ),
            wls_mean_min_v_pu=(
                "wls_min_v_pu",
                "mean",
            ),
            wls_voltage_spread_pu=(
                "wls_min_v_pu",
                lambda x: float(
                    np.max(x)
                    - np.min(x)
                ),
            ),
            wls_max_norm_residual=(
                "wls_max_norm_residual",
                "max",
            ),
            wls_rms_norm_residual=(
                "wls_rms_norm_residual",
                "mean",
            ),
            wls_v_rmse_pu=(
                "wls_v_rmse_pu",
                "mean",
            ),
            truth_min_v_pu=(
                "truth_min_v_pu",
                "min",
            ),
        )
    )

    return agg


def metric_dict(
    y_true,
    y_pred,
):
    y_true = np.asarray(
        y_true,
        dtype=int,
    )
    y_pred = np.asarray(
        y_pred,
        dtype=int,
    )

    neg = (
        y_true == 0
    )

    return {
        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_true,
                y_pred,
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                y_pred,
            )
        ),
        "precision": float(
            precision_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "cyber_detection_rate": float(
            recall_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "physical_false_alarm_rate": float(
            np.mean(
                y_pred[
                    neg
                ] == 1
            )
            if neg.any()
            else np.nan
        ),
    }


def feature_sets():
    return {
        "residual_temporal": [
            "cmd_res_max_pu",
            "cmd_res_rms_pu",
            "cmd_res_fraction_above",
            "cmd_res_longest_run_ms",
            "pref_range_pu",
            "pref_total_variation_pu",
            "pref_max_abs_slope_pu_s",
        ],

        "truth_proxy_fusion": [
            "cmd_res_max_pu",
            "cmd_res_rms_pu",
            "cmd_res_fraction_above",
            "cmd_res_longest_run_ms",
            "network_min_v_pu",
            "network_low_v_fraction_085",
            "pref_range_pu",
            "pref_total_variation_pu",
            "pref_max_abs_slope_pu_s",
        ],

        "wls_fusion": [
            "cmd_res_max_pu",
            "cmd_res_rms_pu",
            "cmd_res_fraction_above",
            "cmd_res_longest_run_ms",
            "wls_min_v_pu",
            "wls_mean_min_v_pu",
            "wls_voltage_spread_pu",
            "wls_max_norm_residual",
            "wls_rms_norm_residual",
            "pref_range_pu",
            "pref_total_variation_pu",
            "pref_max_abs_slope_pu_s",
        ],
    }


def models():
    return {
        "LogisticRegression":
            Pipeline(
                [
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="median"
                        ),
                    ),
                    (
                        "scaler",
                        StandardScaler(),
                    ),
                    (
                        "model",
                        LogisticRegression(
                            max_iter=3000,
                            class_weight="balanced",
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),

        "RandomForest":
            Pipeline(
                [
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="median"
                        ),
                    ),
                    (
                        "model",
                        RandomForestClassifier(
                            n_estimators=600,
                            max_depth=5,
                            min_samples_leaf=2,
                            class_weight="balanced",
                            random_state=RANDOM_STATE,
                            n_jobs=-1,
                        ),
                    ),
                ]
            ),
    }


def select_threshold(
    y_val,
    scores,
):
    thresholds = np.unique(
        np.quantile(
            scores,
            np.linspace(
                0.02,
                0.98,
                120,
            ),
        )
    )

    best = None

    for threshold in thresholds:
        pred = (
            scores
            >= threshold
        ).astype(int)

        m = metric_dict(
            y_val,
            pred,
        )

        utility = (
            m[
                "cyber_detection_rate"
            ]
            - 1.25
            * m[
                "physical_false_alarm_rate"
            ]
        )

        candidate = (
            utility,
            m[
                "balanced_accuracy"
            ],
            -m[
                "physical_false_alarm_rate"
            ],
            float(
                threshold
            ),
        )

        if (
            best is None
            or candidate
            > best
        ):
            best = candidate

    return float(
        best[-1]
    )


def benchmark(
    features,
):
    event = features[
        features[
            "window_mode"
        ] == "event"
    ].copy()

    train = event[
        event[
            "dataset_split"
        ] == "train"
    ]

    val = event[
        event[
            "dataset_split"
        ] == "validation"
    ]

    test = event[
        event[
            "dataset_split"
        ] == "test"
    ]

    rows = []
    predictions = []

    for (
        feature_set,
        cols,
    ) in feature_sets().items():

        missing = [
            c
            for c in cols
            if c not in (
                features.columns
            )
        ]

        if missing:
            raise KeyError(
                f"{feature_set} missing "
                f"columns: {missing}"
            )

        for (
            model_name,
            model,
        ) in models().items():
            model.fit(
                train[
                    cols
                ],
                train[
                    "target_cyber"
                ],
            )

            val_scores = (
                model.predict_proba(
                    val[
                        cols
                    ]
                )[:, 1]
            )

            threshold = (
                select_threshold(
                    val[
                        "target_cyber"
                    ].to_numpy(),
                    val_scores,
                )
            )

            test_scores = (
                model.predict_proba(
                    test[
                        cols
                    ]
                )[:, 1]
            )

            pred = (
                test_scores
                >= threshold
            ).astype(int)

            m = metric_dict(
                test[
                    "target_cyber"
                ],
                pred,
            )

            rows.append(
                {
                    "feature_set":
                        feature_set,
                    "model":
                        model_name,
                    "n_features":
                        len(
                            cols
                        ),
                    "threshold":
                        threshold,
                    **m,
                }
            )

            p = test[
                [
                    "scenario_id",
                    "challenge_type",
                    "target_cyber",
                ]
            ].copy()

            p[
                "feature_set"
            ] = feature_set

            p[
                "model"
            ] = model_name

            p[
                "score"
            ] = test_scores

            p[
                "prediction"
            ] = pred

            predictions.append(
                p
            )

    return (
        pd.DataFrame(
            rows
        ),
        pd.concat(
            predictions,
            ignore_index=True,
        ),
    )


def alarm_by_class(
    predictions,
):
    return (
        predictions.groupby(
            [
                "feature_set",
                "model",
                "challenge_type",
            ],
            as_index=False,
        )
        .agg(
            n=(
                "scenario_id",
                "count",
            ),
            alarm_rate=(
                "prediction",
                "mean",
            ),
        )
    )


def plot_wls_quality(
    snapshots,
):
    fig, ax = plt.subplots(
        figsize=(9.5, 6)
    )

    for challenge, g in (
        snapshots[
            snapshots[
                "window_mode"
            ] == "event"
        ]
        .groupby(
            "challenge_type"
        )
    ):
        ax.scatter(
            g[
                "truth_min_v_pu"
            ],
            g[
                "wls_min_v_pu"
            ],
            s=42,
            alpha=0.75,
            label=challenge,
        )

    lo = float(
        min(
            snapshots[
                "truth_min_v_pu"
            ].min(),
            snapshots[
                "wls_min_v_pu"
            ].min(),
        )
    )

    hi = float(
        max(
            snapshots[
                "truth_min_v_pu"
            ].max(),
            snapshots[
                "wls_min_v_pu"
            ].max(),
        )
    )

    ax.plot(
        [
            lo,
            hi,
        ],
        [
            lo,
            hi,
        ],
        linestyle="--",
        linewidth=1.0,
    )

    ax.set_xlabel(
        "Reconstructed truth min voltage (pu)"
    )
    ax.set_ylabel(
        "WLS-estimated min voltage (pu)"
    )
    ax.set_title(
        "AC WLS Voltage-State Reconstruction on Challenge Snapshots"
    )
    ax.legend(
        frameon=False,
        fontsize=8,
    )

    ax.spines[
        "top"
    ].set_visible(
        False
    )
    ax.spines[
        "right"
    ].set_visible(
        False
    )

    fig.tight_layout()
    fig.savefig(
        FIGURES
        / "47_wls_estimated_vs_truth_voltage.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_detector_comparison(
    metrics,
):
    d = metrics.sort_values(
        [
            "feature_set",
            "model",
        ]
    )

    labels = (
        d[
            "feature_set"
        ]
        + "\n"
        + d[
            "model"
        ]
    ).tolist()

    x = np.arange(
        len(d)
    )

    width = 0.36

    fig, ax = plt.subplots(
        figsize=(12, 6.5)
    )

    ax.bar(
        x - width/2,
        100
        * d[
            "cyber_detection_rate"
        ],
        width,
        label="Cyber detection",
    )

    ax.bar(
        x + width/2,
        100
        * d[
            "physical_false_alarm_rate"
        ],
        width,
        label="Physical false alarm",
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        labels,
        rotation=25,
        ha="right",
        fontsize=8.5,
    )

    ax.set_ylim(
        0,
        105,
    )

    ax.set_ylabel(
        "Held-out test scenario rate (%)"
    )

    ax.set_title(
        "Truth-Proxy vs AC-WLS Physics Fusion"
    )

    ax.legend(
        frameon=False
    )

    ax.spines[
        "top"
    ].set_visible(
        False
    )

    ax.spines[
        "right"
    ].set_visible(
        False
    )

    fig.tight_layout()

    fig.savefig(
        FIGURES
        / "48_truth_vs_wls_fusion_comparison.png",
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_alarm_by_class(
    rates,
):
    d = rates[
        rates[
            "feature_set"
        ] == "wls_fusion"
    ].copy()

    pivot = (
        d.pivot(
            index="challenge_type",
            columns="model",
            values="alarm_rate",
        )
        * 100.0
    )

    fig, ax = plt.subplots(
        figsize=(10.5, 6.2)
    )

    x = np.arange(
        len(
            pivot.index
        )
    )

    models_list = list(
        pivot.columns
    )

    width = (
        0.75
        / max(
            1,
            len(
                models_list
            ),
        )
    )

    for j, model in enumerate(
        models_list
    ):
        offset = (
            j
            - (
                len(
                    models_list
                )
                - 1
            )
            / 2
        ) * width

        ax.bar(
            x + offset,
            pivot[
                model
            ].to_numpy(),
            width,
            label=model,
        )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        pivot.index,
        rotation=20,
        ha="right",
    )

    ax.set_ylim(
        0,
        105,
    )

    ax.set_ylabel(
        "Alarm rate (%)"
    )

    ax.set_title(
        "AC-WLS Fusion Alarm Rate by Ambiguous Challenge Type"
    )

    ax.legend(
        frameon=False
    )

    ax.spines[
        "top"
    ].set_visible(
        False
    )

    ax.spines[
        "right"
    ].set_visible(
        False
    )

    fig.tight_layout()

    fig.savefig(
        FIGURES
        / "49_wls_fusion_alarm_by_challenge.png",
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(fig)


def main():
    for path in [
        CHALLENGE_RAW_PATH,
        CHALLENGE_META_PATH,
        SLOW_CONTEXT_PATH,
        PHASE2N_FEATURE_PATH,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                str(
                    path
                )
            )

    print(
        "\n=== PHASE 2O: AC-WLS ESTIMATED-STATE PHYSICS FUSION ==="
    )

    challenge_raw = pd.read_csv(
        CHALLENGE_RAW_PATH
    )

    challenge_meta = pd.read_csv(
        CHALLENGE_META_PATH,
        parse_dates=[
            "context_timestamp"
        ],
    )

    slow = pd.read_csv(
        SLOW_CONTEXT_PATH,
        parse_dates=[
            "timestamp"
        ],
    )

    phase2n = pd.read_csv(
        PHASE2N_FEATURE_PATH
    )

    snapshots = build_wls_snapshots(
        challenge_raw,
        challenge_meta,
        slow,
    )

    snapshots.to_csv(
        DATA
        / "phase2o_wls_graph_ready_snapshots.csv",
        index=False,
    )

    wls_features = (
        aggregate_wls_features(
            snapshots
        )
    )

    features = (
        phase2n.merge(
            wls_features,
            on=[
                "scenario_id",
                "window_mode",
                "challenge_type",
            ],
            how="left",
            validate="one_to_one",
        )
    )

    # Only keep trajectories with usable WLS outputs.
    before = len(
        features
    )

    features = features.dropna(
        subset=[
            "wls_min_v_pu",
            "wls_mean_min_v_pu",
        ]
    ).copy()

    dropped = (
        before
        - len(
            features
        )
    )

    features.to_csv(
        DATA
        / "phase2o_wls_fusion_feature_dataset.csv",
        index=False,
    )

    print(
        f"\nMerged trajectory rows : "
        f"{len(features)}"
    )

    print(
        f"Dropped for missing WLS: "
        f"{dropped}"
    )

    event_features = features[
        features[
            "window_mode"
        ] == "event"
    ]

    print(
        "\nWLS voltage-estimation quality:"
    )

    print(
        event_features.groupby(
            "challenge_type"
        )[
            [
                "wls_v_rmse_pu",
                "wls_max_norm_residual",
            ]
        ].agg(
            [
                "mean",
                "max",
            ]
        ).to_string()
    )

    metrics, predictions = (
        benchmark(
            features
        )
    )

    rates = alarm_by_class(
        predictions
    )

    metrics.to_csv(
        RESULTS
        / "phase2o_wls_fusion_test_metrics.csv",
        index=False,
    )

    rates.to_csv(
        RESULTS
        / "phase2o_wls_fusion_alarm_by_challenge.csv",
        index=False,
    )

    plot_wls_quality(
        snapshots
    )

    plot_detector_comparison(
        metrics
    )

    plot_alarm_by_class(
        rates
    )

    print(
        "\n=== HELD-OUT TEST: TRUTH vs WLS FUSION ==="
    )

    print(
        metrics[
            [
                "feature_set",
                "model",
                "balanced_accuracy",
                "f1",
                "precision",
                "cyber_detection_rate",
                "physical_false_alarm_rate",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\n=== WLS FUSION ALARM RATE BY CHALLENGE ==="
    )

    print(
        rates[
            rates[
                "feature_set"
            ] == "wls_fusion"
        ].to_string(
            index=False
        )
    )

    print(
        "\nSaved:"
    )

    print(
        "  data/phase2o_wls_graph_ready_snapshots.csv"
    )

    print(
        "  data/phase2o_wls_fusion_feature_dataset.csv"
    )

    print(
        "  results/phase2o_wls_fusion_test_metrics.csv"
    )

    print(
        "  results/phase2o_wls_fusion_alarm_by_challenge.csv"
    )

    print(
        "  figures/47_wls_estimated_vs_truth_voltage.png"
    )

    print(
        "  figures/48_truth_vs_wls_fusion_comparison.png"
    )

    print(
        "  figures/49_wls_fusion_alarm_by_challenge.png"
    )

    print(
        "\nINTERPRETATION RULE:\n"
        "The truth-proxy fusion from Phase 2N was an optimistic diagnostic. "
        "This phase replaces its key network-voltage context with a noisy, "
        "redundant AC WLS estimate. If WLS fusion preserves low physical "
        "false alarms and useful mild-current-limit detection, the network "
        "context survives a realistic estimation layer. The graph-ready "
        "snapshot file also stores estimated bus voltages/angles/injections "
        "and line P/Q flows. Only after this succeeds should topology-aware "
        "localization or a GNN be introduced. If WLS fusion degrades badly, "
        "the next work should improve observability/measurement placement "
        "rather than add a more complex classifier."
    )


if __name__ == "__main__":
    main()
