from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    recall_score,
)

warnings.filterwarnings("ignore")


# ============================================================
# PHASE 2S
# TOPOLOGY-AWARE CYBER-vs-PHYSICAL DETECTION + LOCALIZATION
# ============================================================
#
# Purpose
# -------
# Phase 2R established graph-ready WLS + protected-prior telemetry.
# This phase asks a deliberately narrow scientific question:
#
#   Does explicit topology-aware relational evidence improve over
#   residual-only and prior-only baselines for:
#
#   (A) cyber-vs-benign detection, where legitimate physical load
#       disturbances are hard negatives, and
#   (B) attack-target bus localization?
#
# IMPORTANT:
# - No GNN is used here.
# - This is the interpretable pre-GNN ablation.
# - 'eval_*' and 'label_*' columns are NEVER model features.
# - Splits remain replication-group safe from Phase 2R.
#
# If topology-fusion does not improve meaningful baselines, a GNN
# is not justified yet. If it does, Phase 2T can test whether graph
# learning improves further.
# ============================================================


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

NODE_PATH = DATA / "phase2r_hard_graph_node_features.csv"
EDGE_PATH = DATA / "phase2r_hard_graph_edge_features.csv"
META_PATH = DATA / "phase2r_hard_graph_metadata.csv"

RANDOM_SEED = 20260812

CYBER_CASES = {
    "naive_single_sensor_corruption",
    "nonlinear_model_consistent_fdia",
}

BENIGN_CASES = {
    "clean_noisy",
    "physical_load_disturbance",
}


def safe_ratio(a, b, eps=1e-9):
    return float(a) / (float(b) + eps)


def load_inputs():
    for p in [NODE_PATH, EDGE_PATH, META_PATH]:
        if not p.exists():
            raise FileNotFoundError(
                f"Missing required Phase-2R file: {p}"
            )

    node = pd.read_csv(NODE_PATH)
    edge = pd.read_csv(EDGE_PATH)
    meta = pd.read_csv(META_PATH)

    if "context_timestamp" in node.columns:
        node["context_timestamp"] = pd.to_datetime(
            node["context_timestamp"]
        )
    if "context_timestamp" in meta.columns:
        meta["context_timestamp"] = pd.to_datetime(
            meta["context_timestamp"]
        )

    return node, edge, meta


def add_topology_relational_node_features(node, edge):
    """
    Build local neighborhood and incident-edge features without using
    any attack labels/evaluation truth.
    """
    out = node.copy()

    # Build graph-specific adjacency from edge table.
    edge_by_graph = {
        gid: g.copy()
        for gid, g in edge.groupby("graph_id")
    }

    derived_rows = []

    for gid, gnodes in out.groupby("graph_id"):
        gedges = edge_by_graph[gid]
        local = gnodes.set_index("node_id")

        adjacency = {int(n): set() for n in local.index}
        for _, er in gedges.iterrows():
            f = int(er["from_bus"])
            t = int(er["to_bus"])
            adjacency[f].add(t)
            adjacency[t].add(f)

        for node_id, nr in local.iterrows():
            nbs = sorted(adjacency[int(node_id)])

            nb_inv = local.loc[nbs, "innovation_score"].to_numpy(
                dtype=float
            )
            nb_res = local.loc[nbs, "residual_score"].to_numpy(
                dtype=float
            )

            incident = gedges[
                (gedges["from_bus"] == int(node_id))
                | (gedges["to_bus"] == int(node_id))
            ]

            inc_inv = incident["innovation_score"].to_numpy(dtype=float)
            inc_res = incident["residual_score"].to_numpy(dtype=float)

            local_inv = float(nr["innovation_score"])
            local_res = float(nr["residual_score"])

            derived_rows.append({
                "graph_id": gid,
                "node_id": int(node_id),

                "neighbor_mean_innovation": float(np.mean(nb_inv)),
                "neighbor_max_innovation": float(np.max(nb_inv)),
                "neighbor_mean_residual": float(np.mean(nb_res)),
                "neighbor_max_residual": float(np.max(nb_res)),

                "incident_edge_mean_innovation": float(
                    np.mean(inc_inv)
                ),
                "incident_edge_max_innovation": float(
                    np.max(inc_inv)
                ),
                "incident_edge_mean_residual": float(
                    np.mean(inc_res)
                ),
                "incident_edge_max_residual": float(
                    np.max(inc_res)
                ),

                "local_minus_neighbor_innovation": (
                    local_inv - float(np.mean(nb_inv))
                ),
                "local_over_neighbor_innovation": safe_ratio(
                    local_inv, np.mean(nb_inv)
                ),
                "local_minus_neighbor_residual": (
                    local_res - float(np.mean(nb_res))
                ),
                "local_over_neighbor_residual": safe_ratio(
                    local_res, np.mean(nb_res)
                ),

                "local_minus_incident_edge_innovation": (
                    local_inv - float(np.mean(inc_inv))
                ),
            })

    derived = pd.DataFrame(derived_rows)

    out = out.merge(
        derived,
        on=["graph_id", "node_id"],
        how="left",
        validate="one_to_one",
    )

    return out


def build_graph_features(node, edge, meta):
    rows = []

    for gid, m in meta.set_index("graph_id").iterrows():
        n = node[node["graph_id"] == gid].copy()
        e = edge[edge["graph_id"] == gid].copy()

        if len(n) != 5 or len(e) != 4:
            raise RuntimeError(
                f"{gid}: expected 5 nodes / 4 edges, "
                f"got {len(n)} / {len(e)}"
            )

        n = n.sort_values("node_id")
        e = e.sort_values("edge_id")

        r = {
            "graph_id": gid,
            "replication": int(m["replication"]),
            "dataset_split": str(m["dataset_split"]),
            "case": str(m["case"]),
            "family": str(m["family"]),
            "is_cyber": int(str(m["case"]) in CYBER_CASES),

            # Graph-level residual baseline.
            "bdd_j_ratio": safe_ratio(
                m["chi2_j"], m["chi2_threshold"]
            ),
            "max_abs_norm_residual": float(
                m["max_abs_norm_residual"]
            ),
        }

        # ----------------------------
        # Fixed-node features
        # ----------------------------
        for _, nr in n.iterrows():
            b = int(nr["node_id"])

            # Residual-only features
            r[f"b{b}_residual_score"] = float(
                nr["residual_score"]
            )
            r[f"b{b}_res_v"] = float(nr["norm_residual_v"])
            r[f"b{b}_res_p"] = float(nr["norm_residual_p"])
            r[f"b{b}_res_q"] = float(nr["norm_residual_q"])

            # Prior innovation features
            r[f"b{b}_innovation_score"] = float(
                nr["innovation_score"]
            )
            r[f"b{b}_dvm"] = float(nr["innovation_vm_pu"])
            r[f"b{b}_dva"] = float(nr["innovation_va_deg"])
            r[f"b{b}_dp"] = float(nr["innovation_p_mw"])
            r[f"b{b}_dq"] = float(nr["innovation_q_mvar"])

            # Explicit topology-aware relational features
            for c in [
                "neighbor_mean_innovation",
                "neighbor_max_innovation",
                "neighbor_mean_residual",
                "neighbor_max_residual",
                "incident_edge_mean_innovation",
                "incident_edge_max_innovation",
                "incident_edge_mean_residual",
                "incident_edge_max_residual",
                "local_minus_neighbor_innovation",
                "local_over_neighbor_innovation",
                "local_minus_neighbor_residual",
                "local_over_neighbor_residual",
                "local_minus_incident_edge_innovation",
            ]:
                r[f"b{b}_{c}"] = float(nr[c])

        # ----------------------------
        # Fixed-edge features
        # ----------------------------
        for _, er in e.iterrows():
            k = int(er["edge_id"])

            r[f"e{k}_residual_score"] = float(
                er["residual_score"]
            )
            r[f"e{k}_innovation_score"] = float(
                er["innovation_score"]
            )
            r[f"e{k}_dp"] = float(
                er["innovation_p_from_mw"]
            )
            r[f"e{k}_dq"] = float(
                er["innovation_q_from_mvar"]
            )

        # ----------------------------
        # Permutation-invariant global summaries
        # ----------------------------
        inv = n["innovation_score"].to_numpy(dtype=float)
        res = n["residual_score"].to_numpy(dtype=float)
        einv = e["innovation_score"].to_numpy(dtype=float)
        eres = e["residual_score"].to_numpy(dtype=float)

        inv_sorted = np.sort(inv)[::-1]
        res_sorted = np.sort(res)[::-1]

        r.update({
            "node_inv_mean": float(np.mean(inv)),
            "node_inv_std": float(np.std(inv)),
            "node_inv_max": float(np.max(inv)),
            "node_inv_second": float(inv_sorted[1]),
            "node_inv_peak_to_mean": safe_ratio(
                np.max(inv), np.mean(inv)
            ),
            "node_inv_peak_to_second": safe_ratio(
                inv_sorted[0], inv_sorted[1]
            ),

            "node_res_mean": float(np.mean(res)),
            "node_res_std": float(np.std(res)),
            "node_res_max": float(np.max(res)),
            "node_res_second": float(res_sorted[1]),
            "node_res_peak_to_mean": safe_ratio(
                np.max(res), np.mean(res)
            ),

            "edge_inv_mean": float(np.mean(einv)),
            "edge_inv_max": float(np.max(einv)),
            "edge_res_mean": float(np.mean(eres)),
            "edge_res_max": float(np.max(eres)),
        })

        # Device-role aggregate features. These are structural,
        # not labels.
        load_mask = n["role_load"].astype(int) == 1
        pv_mask = n["role_pv_gfl"].astype(int) == 1
        bess_mask = n["role_bess_gfm"].astype(int) == 1

        r["load_bus_inv_max"] = float(
            n.loc[load_mask, "innovation_score"].max()
        )
        r["load_bus_res_max"] = float(
            n.loc[load_mask, "residual_score"].max()
        )
        r["pv_bus_inv"] = float(
            n.loc[pv_mask, "innovation_score"].iloc[0]
        )
        r["bess_bus_inv"] = float(
            n.loc[bess_mask, "innovation_score"].iloc[0]
        )

        rows.append(r)

    return pd.DataFrame(rows)


def feature_columns(graph_df):
    metadata = {
        "graph_id",
        "replication",
        "dataset_split",
        "case",
        "family",
        "is_cyber",
    }

    all_features = [
        c for c in graph_df.columns
        if c not in metadata
        and not c.startswith("eval_")
        and not c.startswith("label_")
    ]

    # Explicit topology-relational columns (neighbor/incident-edge
    # comparisons) are named e.g. "b0_neighbor_mean_residual" or
    # "b0_incident_edge_max_innovation" -- they contain the substrings
    # "residual"/"innovation" like the plain per-bus features do, so
    # they must be excluded here by name, or they silently leak into
    # residual_only/prior_only and those stop being topology-free
    # baselines. (Caught empirically: before this exclusion, all 65
    # relational columns in topology_fusion were double-counted this
    # way -- 30 into residual_only, 35 into prior_only -- so that
    # residual_only | prior_only == topology_fusion exactly, i.e.
    # topology_fusion added no information beyond the two "baselines"
    # combined.)
    relational_markers = (
        "neighbor_",
        "incident_edge_",
        "local_minus_",
        "local_over_",
    )

    def is_relational(c):
        return any(mk in c for mk in relational_markers)

    residual_only = [
        c for c in all_features
        if (
            "residual" in c
            or "_res_" in c
            or c.startswith("bdd_")
            or c.startswith("node_res_")
            or c.startswith("edge_res_")
            or c.endswith("_res_max")
        )
        and "innovation" not in c
        and not is_relational(c)
    ]

    prior_only = [
        c for c in all_features
        if (
            "innovation" in c
            or c.endswith("_dvm")
            or c.endswith("_dva")
            or c.endswith("_dp")
            or c.endswith("_dq")
            or c.startswith("node_inv_")
            or c.startswith("edge_inv_")
            or c.endswith("_bus_inv_max")
            or c in {"pv_bus_inv", "bess_bus_inv"}
        )
        and "residual" not in c
        and not is_relational(c)
    ]

    topology_fusion = all_features

    return {
        "residual_only": sorted(set(residual_only)),
        "prior_only": sorted(set(prior_only)),
        "residual_plus_prior": sorted(set(residual_only) | set(prior_only)),
        "topology_fusion": sorted(set(topology_fusion)),
    }


def make_models(seed):
    return {
        "LogisticRegression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                random_state=seed,
            )),
        ]),
        "RandomForest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(
                n_estimators=600,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=seed,
                n_jobs=-1,
            )),
        ]),
        "HistGradientBoosting": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingClassifier(
                max_iter=300,
                learning_rate=0.06,
                max_leaf_nodes=15,
                l2_regularization=1.0,
                random_state=seed,
            )),
        ]),
    }


def best_threshold(y, scores, case_names):
    """
    Select threshold on validation data only.
    Primary objective: balanced accuracy.
    Tie-breaks: lower physical false alarm, then higher cyber recall.
    """
    y = np.asarray(y, dtype=int)
    scores = np.asarray(scores, dtype=float)
    case_names = np.asarray(case_names, dtype=str)

    candidates = np.unique(np.r_[
        np.quantile(scores, np.linspace(0.0, 1.0, 201)),
        0.5,
    ])

    best = None

    for t in candidates:
        pred = (scores >= t).astype(int)
        bal = balanced_accuracy_score(y, pred)

        physical_mask = (
            case_names == "physical_load_disturbance"
        )
        cyber_mask = y == 1

        physical_fpr = (
            float(np.mean(pred[physical_mask]))
            if physical_mask.any() else 0.0
        )
        cyber_recall = (
            float(np.mean(pred[cyber_mask]))
            if cyber_mask.any() else 0.0
        )

        key = (
            bal,
            -physical_fpr,
            cyber_recall,
        )

        if best is None or key > best["key"]:
            best = {
                "threshold": float(t),
                "key": key,
            }

    return best["threshold"]


def detection_metrics(
    y,
    scores,
    threshold,
    cases,
):
    y = np.asarray(y, dtype=int)
    scores = np.asarray(scores, dtype=float)
    cases = np.asarray(cases, dtype=str)
    pred = (scores >= threshold).astype(int)

    cyber_mask = y == 1
    benign_mask = y == 0
    physical_mask = cases == "physical_load_disturbance"
    clean_mask = cases == "clean_noisy"
    stealth_mask = cases == "nonlinear_model_consistent_fdia"
    naive_mask = cases == "naive_single_sensor_corruption"

    return {
        "roc_auc": float(roc_auc_score(y, scores)),
        "average_precision": float(
            average_precision_score(y, scores)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(y, pred)
        ),
        "f1": float(f1_score(y, pred)),
        "cyber_recall": float(np.mean(pred[cyber_mask])),
        "benign_fpr": float(np.mean(pred[benign_mask])),
        "clean_fpr": float(np.mean(pred[clean_mask])),
        "physical_fpr": float(np.mean(pred[physical_mask])),
        "stealth_recall": float(np.mean(pred[stealth_mask])),
        "naive_recall": float(np.mean(pred[naive_mask])),
    }


def bootstrap_group_ci(
    test_df,
    scores,
    threshold,
    n_boot=1200,
    seed=RANDOM_SEED,
):
    """
    Bootstrap by replication, preserving the four matched cases.
    """
    rng = np.random.default_rng(seed)

    tmp = test_df[
        ["replication", "case", "is_cyber"]
    ].copy()
    tmp["score"] = np.asarray(scores, dtype=float)

    reps = tmp["replication"].unique()
    values = []

    for _ in range(n_boot):
        sampled = rng.choice(reps, size=len(reps), replace=True)
        parts = []

        for j, rep in enumerate(sampled):
            g = tmp[tmp["replication"] == rep].copy()
            g["boot_rep"] = j
            parts.append(g)

        b = pd.concat(parts, ignore_index=True)

        m = detection_metrics(
            b["is_cyber"].to_numpy(),
            b["score"].to_numpy(),
            threshold,
            b["case"].to_numpy(),
        )

        values.append([
            m["balanced_accuracy"],
            m["f1"],
            m["cyber_recall"],
            m["physical_fpr"],
            m["stealth_recall"],
        ])

    arr = np.asarray(values, dtype=float)
    names = [
        "balanced_accuracy",
        "f1",
        "cyber_recall",
        "physical_fpr",
        "stealth_recall",
    ]

    out = {}

    for j, name in enumerate(names):
        out[f"{name}_ci_low"] = float(
            np.quantile(arr[:, j], 0.025)
        )
        out[f"{name}_ci_high"] = float(
            np.quantile(arr[:, j], 0.975)
        )

    return out


def run_simple_detection_baseline(
    graph_df,
    score_column,
    name,
):
    train = graph_df[graph_df["dataset_split"] == "train"]
    val = graph_df[graph_df["dataset_split"] == "validation"]
    test = graph_df[graph_df["dataset_split"] == "test"]

    # No fitting required; threshold selected on validation only.
    threshold = best_threshold(
        val["is_cyber"],
        val[score_column],
        val["case"],
    )

    m = detection_metrics(
        test["is_cyber"],
        test[score_column],
        threshold,
        test["case"],
    )
    ci = bootstrap_group_ci(
        test,
        test[score_column],
        threshold,
    )

    return {
        "feature_set": name,
        "model": "InterpretableScore",
        "n_features": 1,
        "threshold": threshold,
        **m,
        **ci,
    }


def run_learned_detection(graph_df, feat_sets):
    train = graph_df[graph_df["dataset_split"] == "train"].copy()
    val = graph_df[graph_df["dataset_split"] == "validation"].copy()
    test = graph_df[graph_df["dataset_split"] == "test"].copy()

    rows = []
    fitted = {}

    models = make_models(RANDOM_SEED)

    for fs_name, cols in feat_sets.items():
        for model_name, model in models.items():
            mdl = clone(model)

            mdl.fit(
                train[cols],
                train["is_cyber"],
            )

            val_scores = mdl.predict_proba(
                val[cols]
            )[:, 1]

            threshold = best_threshold(
                val["is_cyber"],
                val_scores,
                val["case"],
            )

            test_scores = mdl.predict_proba(
                test[cols]
            )[:, 1]

            m = detection_metrics(
                test["is_cyber"],
                test_scores,
                threshold,
                test["case"],
            )

            ci = bootstrap_group_ci(
                test,
                test_scores,
                threshold,
            )

            val_pred = (val_scores >= threshold).astype(int)
            val_bal = balanced_accuracy_score(
                val["is_cyber"],
                val_pred,
            )

            rows.append({
                "feature_set": fs_name,
                "model": model_name,
                "n_features": len(cols),
                "threshold": threshold,
                "validation_balanced_accuracy": float(val_bal),
                **m,
                **ci,
            })

            fitted[(fs_name, model_name)] = {
                "model": mdl,
                "features": cols,
                "threshold": threshold,
                "test_scores": test_scores,
            }

    return pd.DataFrame(rows), fitted


def build_node_learning_table(node):
    out = node.copy()

    out["is_cyber_graph"] = out["case"].isin(
        CYBER_CASES
    ).astype(int)

    out["target_node"] = (
        (out["is_cyber_graph"] == 1)
        & (out["label_is_attack_target_node"] == 1)
    ).astype(int)

    return out


def node_feature_sets(node):
    # FIX (caught in post-submission audit): "degree" and the four
    # "role_*" one-hot flags used to be prepended to every feature set
    # below, including residual_only/prior_only/residual_plus_prior.
    # "degree" is graph-topology information -- it does not belong in
    # anything called non-relational/topology-free. "role_load" is
    # worse: the attack generator only ever targets load buses
    # (target_bus = rng.choice(load_buses) in the scenario generator),
    # so handing the localizer role_load is close to handing it the
    # answer's eligible-candidate set directly, regardless of which
    # feature set is nominally being tested. Dropped from all four
    # sets below (including topology_fusion -- this structural/role
    # metadata was never part of what this paper calls "topology-
    # relational": the neighbor/incident-edge comparison features
    # below already cover that).
    residual = [
        "norm_residual_v",
        "norm_residual_p",
        "norm_residual_q",
        "abs_norm_residual_v",
        "abs_norm_residual_p",
        "abs_norm_residual_q",
        "residual_score",
    ]

    prior = [
        "innovation_vm_pu",
        "innovation_va_deg",
        "innovation_p_mw",
        "innovation_q_mvar",
        "innovation_score",
    ]

    topology = sorted(set(
        residual
        + prior
        + [
            "neighbor_mean_innovation",
            "neighbor_max_innovation",
            "neighbor_mean_residual",
            "neighbor_max_residual",
            "incident_edge_mean_innovation",
            "incident_edge_max_innovation",
            "incident_edge_mean_residual",
            "incident_edge_max_residual",
            "local_minus_neighbor_innovation",
            "local_over_neighbor_innovation",
            "local_minus_neighbor_residual",
            "local_over_neighbor_residual",
            "local_minus_incident_edge_innovation",
        ]
    ))

    return {
        "residual_only": residual,
        "prior_only": prior,
        "residual_plus_prior": sorted(set(residual) | set(prior)),
        "topology_fusion": topology,
    }


def node_models(seed):
    return {
        "LogisticRegression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                random_state=seed,
            )),
        ]),
        "RandomForest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(
                n_estimators=700,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=seed,
                n_jobs=-1,
            )),
        ]),
        "HistGradientBoosting": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingClassifier(
                max_iter=300,
                learning_rate=0.06,
                max_leaf_nodes=15,
                l2_regularization=1.0,
                random_state=seed,
            )),
        ]),
    }


def rank_localization_metrics(df, scores):
    tmp = df[
        [
            "graph_id",
            "case",
            "node_id",
            "target_node",
            "label_is_physical_target_node",
        ]
    ].copy()
    tmp["score"] = np.asarray(scores, dtype=float)

    cyber = tmp[tmp["case"].isin(CYBER_CASES)].copy()
    physical = tmp[
        tmp["case"] == "physical_load_disturbance"
    ].copy()

    top1 = []
    top2 = []
    rr = []
    by_case = {c: [] for c in CYBER_CASES}

    for gid, g in cyber.groupby("graph_id"):
        g = g.sort_values("score", ascending=False).reset_index(drop=True)
        target_positions = np.flatnonzero(
            g["target_node"].to_numpy() == 1
        )

        if len(target_positions) != 1:
            continue

        rank = int(target_positions[0]) + 1
        hit1 = int(rank == 1)

        top1.append(hit1)
        top2.append(int(rank <= 2))
        rr.append(1.0 / rank)

        case_name = str(g["case"].iloc[0])
        by_case[case_name].append(hit1)

    # Diagnostic only: does a physical disturbance also point strongly
    # to its true disturbed load bus? This quantifies ambiguity.
    physical_top1 = []

    for gid, g in physical.groupby("graph_id"):
        g = g.sort_values("score", ascending=False)
        pred_node = int(g.iloc[0]["node_id"])
        target_rows = g[
            g["label_is_physical_target_node"] == 1
        ]
        if len(target_rows) == 1:
            target_node = int(target_rows.iloc[0]["node_id"])
            physical_top1.append(int(pred_node == target_node))

    return {
        "top1_accuracy": float(np.mean(top1)),
        "top2_accuracy": float(np.mean(top2)),
        "mean_reciprocal_rank": float(np.mean(rr)),
        "stealth_top1_accuracy": float(
            np.mean(
                by_case["nonlinear_model_consistent_fdia"]
            )
        ),
        "naive_top1_accuracy": float(
            np.mean(
                by_case["naive_single_sensor_corruption"]
            )
        ),
        "physical_target_top1_diagnostic": float(
            np.mean(physical_top1)
        ),
    }


def run_simple_localization(node):
    test = node[node["dataset_split"] == "test"].copy()

    rows = []

    for name, col in [
        ("ResidualArgmax", "residual_score"),
        ("PriorInnovationArgmax", "innovation_score"),
    ]:
        m = rank_localization_metrics(
            test,
            test[col].to_numpy(),
        )
        rows.append({
            "feature_set": (
                "residual_only"
                if name == "ResidualArgmax"
                else "prior_only"
            ),
            "model": name,
            "n_features": 1,
            **m,
        })

    return rows


def run_learned_localization(node, feat_sets):
    train = node[node["dataset_split"] == "train"].copy()
    val = node[node["dataset_split"] == "validation"].copy()
    test = node[node["dataset_split"] == "test"].copy()

    rows = []
    models = node_models(RANDOM_SEED)

    for fs_name, cols in feat_sets.items():
        for model_name, model in models.items():
            mdl = clone(model)

            if model_name == "HistGradientBoosting":
                ytr = train["target_node"].to_numpy(dtype=int)
                w = np.where(
                    ytr == 1,
                    max(
                        1.0,
                        (len(ytr) - ytr.sum())
                        / max(ytr.sum(), 1)
                    ),
                    1.0,
                )
                mdl.fit(
                    train[cols],
                    train["target_node"],
                    model__sample_weight=w,
                )
            else:
                mdl.fit(
                    train[cols],
                    train["target_node"],
                )

            val_scores = mdl.predict_proba(
                val[cols]
            )[:, 1]
            val_m = rank_localization_metrics(
                val,
                val_scores,
            )

            test_scores = mdl.predict_proba(
                test[cols]
            )[:, 1]
            test_m = rank_localization_metrics(
                test,
                test_scores,
            )

            rows.append({
                "feature_set": fs_name,
                "model": model_name,
                "n_features": len(cols),
                "validation_top1_accuracy": (
                    val_m["top1_accuracy"]
                ),
                **test_m,
            })

    return pd.DataFrame(rows)


def plot_detection(metrics):
    d = metrics.copy()

    # Keep a readable subset: simple baselines + best learned result
    simple = d[d["model"] == "InterpretableScore"].copy()

    learned = d[d["model"] != "InterpretableScore"].copy()
    best_idx = learned["balanced_accuracy"].idxmax()
    best = learned.loc[[best_idx]]

    show = pd.concat([simple, best], ignore_index=True)
    labels = (
        show["feature_set"] + "\n" + show["model"]
    ).tolist()

    x = np.arange(len(show))
    width = 0.36

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(
        x - width / 2,
        100 * show["cyber_recall"],
        width,
        label="Cyber recall",
    )
    ax.bar(
        x + width / 2,
        100 * show["physical_fpr"],
        width,
        label="Physical false alarm",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=12, ha="right")
    ax.set_ylabel("Rate (%)")
    ax.set_ylim(0, 105)
    ax.set_title(
        "Cyber detection with physical disturbances as hard negatives"
    )
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(
        FIGURES / "51_topology_detection_tradeoff_hard.png",
        dpi=220,
    )
    plt.close(fig)


def plot_localization(metrics):
    d = metrics.copy().sort_values(
        "top1_accuracy",
        ascending=False,
    )

    # Keep simple baselines and best topology-fusion learned result.
    simple = d[d["model"].isin([
        "ResidualArgmax",
        "PriorInnovationArgmax",
    ])]

    topo = d[
        (d["feature_set"] == "topology_fusion")
        & (~d["model"].isin([
            "ResidualArgmax",
            "PriorInnovationArgmax",
        ]))
    ]

    if not topo.empty:
        topo = topo.head(1)

    show = pd.concat([simple, topo], ignore_index=True)

    labels = (
        show["feature_set"] + "\n" + show["model"]
    ).tolist()

    x = np.arange(len(show))

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(
        x,
        100 * show["top1_accuracy"],
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=12, ha="right")
    ax.set_ylabel("Top-1 attack-bus localization (%)")
    ax.set_ylim(0, 105)
    ax.set_title(
        "Attack-target localization on held-out cyber graphs"
    )
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(
        FIGURES / "52_topology_localization_top1_hard.png",
        dpi=220,
    )
    plt.close(fig)


def plot_score_map(graph_df):
    fig, ax = plt.subplots(figsize=(8.5, 6))

    for case_name, g in graph_df.groupby("case"):
        ax.scatter(
            g["node_res_max"],
            g["node_inv_max"],
            alpha=0.65,
            label=case_name,
        )

    ax.set_xlabel("Max node WLS residual score")
    ax.set_ylabel("Max node protected-prior innovation score")
    ax.set_title(
        "Residual consistency vs protected-prior inconsistency"
    )
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(
        FIGURES / "53_residual_vs_prior_graph_map_hard.png",
        dpi=220,
    )
    plt.close(fig)


def save_rf_importance(graph_df, feat_sets):
    train = graph_df[
        graph_df["dataset_split"] == "train"
    ].copy()

    cols = feat_sets["topology_fusion"]

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(
            n_estimators=900,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )),
    ])

    pipe.fit(
        train[cols],
        train["is_cyber"],
    )

    imp = pipe.named_steps["model"].feature_importances_

    out = pd.DataFrame({
        "feature": cols,
        "importance": imp,
    }).sort_values("importance", ascending=False)

    out.to_csv(
        RESULTS / "phase2s_hard_topology_rf_feature_importance.csv",
        index=False,
    )

    top = out.head(20).sort_values("importance")

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(top["feature"], top["importance"])
    ax.set_xlabel("Random Forest feature importance")
    ax.set_title("Topology-fusion diagnostic feature ranking")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(
        FIGURES / "54_topology_fusion_feature_importance_hard.png",
        dpi=220,
    )
    plt.close(fig)

    return out


def main():
    node, edge, meta = load_inputs()

    node = add_topology_relational_node_features(
        node,
        edge,
    )

    graph_df = build_graph_features(
        node,
        edge,
        meta,
    )

    feat_sets = feature_columns(graph_df)

    print(
        "\n=== PHASE 2S: TOPOLOGY-AWARE CYBER-vs-PHYSICAL + "
        "LOCALIZATION ==="
    )
    print(f"Graph snapshots : {len(graph_df)}")
    print(f"Node rows       : {len(node)}")
    print(f"Edge rows       : {len(edge)}")

    print("\nGraph split counts:")
    print(
        graph_df.groupby(
            ["dataset_split", "case"]
        ).size().unstack(fill_value=0).to_string()
    )

    print("\nDetection feature counts:")
    for k, v in feat_sets.items():
        print(f"  {k:16s}: {len(v)}")

    # ========================================================
    # Detection
    # ========================================================
    detection_rows = []

    detection_rows.append(
        run_simple_detection_baseline(
            graph_df,
            "node_res_max",
            "residual_only",
        )
    )

    detection_rows.append(
        run_simple_detection_baseline(
            graph_df,
            "node_inv_max",
            "prior_only",
        )
    )

    learned_df, fitted = run_learned_detection(
        graph_df,
        feat_sets,
    )

    # FIX (caught in post-submission audit): run_learned_detection()
    # fits and test-scores all 3 candidate models per feature set, and
    # validation_balanced_accuracy was already computed for each but
    # never used to pick a winner -- downstream code (26_multi_seed_
    # replication.py, 29_paper_figures.py) instead took the max TEST
    # balanced_accuracy across the 3 models, which is test-set model
    # selection (optimistic bias), the same bug independently found
    # and fixed in 28_ieee14_scale_replication.py. Now: keep exactly
    # one (validation-selected) model's row per feature set here, so
    # every downstream .max()/.groupby() over this file's feature_set
    # column is a no-op over an already-selected value, not a selection
    # itself.
    learned_df = (
        learned_df
        .sort_values("validation_balanced_accuracy", ascending=False)
        .drop_duplicates(subset="feature_set", keep="first")
    )

    detection_df = pd.concat(
        [
            pd.DataFrame(detection_rows),
            learned_df,
        ],
        ignore_index=True,
        sort=False,
    )

    detection_df = detection_df.sort_values(
        [
            "balanced_accuracy",
            "physical_fpr",
            "stealth_recall",
        ],
        ascending=[False, True, False],
    )

    # ========================================================
    # Localization
    # ========================================================
    node_learn = build_node_learning_table(node)
    node_sets = node_feature_sets(node_learn)

    loc_rows = run_simple_localization(node_learn)
    learned_loc = run_learned_localization(
        node_learn,
        node_sets,
    )

    # FIX (caught in post-submission audit): same test-set model-
    # selection bug as run_learned_detection() above, for localization.
    learned_loc = (
        learned_loc
        .sort_values("validation_top1_accuracy", ascending=False)
        .drop_duplicates(subset="feature_set", keep="first")
    )

    localization_df = pd.concat(
        [
            pd.DataFrame(loc_rows),
            learned_loc,
        ],
        ignore_index=True,
        sort=False,
    )

    localization_df = localization_df.sort_values(
        [
            "top1_accuracy",
            "stealth_top1_accuracy",
        ],
        ascending=False,
    )

    # ========================================================
    # Save
    # ========================================================
    graph_df.to_csv(
        DATA / "phase2s_hard_graph_feature_matrix.csv",
        index=False,
    )

    node_learn.to_csv(
        DATA / "phase2s_hard_node_relational_feature_matrix.csv",
        index=False,
    )

    detection_df.to_csv(
        RESULTS / "phase2s_hard_cyber_detection_metrics.csv",
        index=False,
    )

    localization_df.to_csv(
        RESULTS / "phase2s_hard_attack_localization_metrics.csv",
        index=False,
    )

    importance = save_rf_importance(
        graph_df,
        feat_sets,
    )

    plot_detection(detection_df)
    plot_localization(localization_df)
    plot_score_map(graph_df)

    # ========================================================
    # Print
    # ========================================================
    print("\n=== CYBER-vs-PHYSICAL DETECTION — TEST ===")
    cols = [
        "feature_set",
        "model",
        "n_features",
        "balanced_accuracy",
        "f1",
        "cyber_recall",
        "physical_fpr",
        "clean_fpr",
        "stealth_recall",
        "naive_recall",
        "roc_auc",
    ]
    print(
        detection_df[cols]
        .head(12)
        .to_string(index=False)
    )

    print("\n95% replication-bootstrap CI for top detection rows:")
    ci_cols = [
        "feature_set",
        "model",
        "balanced_accuracy_ci_low",
        "balanced_accuracy_ci_high",
        "stealth_recall_ci_low",
        "stealth_recall_ci_high",
        "physical_fpr_ci_low",
        "physical_fpr_ci_high",
    ]
    print(
        detection_df[ci_cols]
        .head(6)
        .to_string(index=False)
    )

    print("\n=== ATTACK-BUS LOCALIZATION — TEST ===")
    loc_cols = [
        "feature_set",
        "model",
        "n_features",
        "top1_accuracy",
        "top2_accuracy",
        "mean_reciprocal_rank",
        "stealth_top1_accuracy",
        "naive_top1_accuracy",
        "physical_target_top1_diagnostic",
    ]
    print(
        localization_df[loc_cols]
        .head(12)
        .to_string(index=False)
    )

    best_det = detection_df.iloc[0]
    best_loc = localization_df.iloc[0]

    print("\n=== SCIENTIFIC DECISION ===")
    print(
        f"Best detection : {best_det['feature_set']} / "
        f"{best_det['model']} | "
        f"BalAcc={best_det['balanced_accuracy']:.3f} | "
        f"Stealth recall={best_det['stealth_recall']:.3f} | "
        f"Physical FPR={best_det['physical_fpr']:.3f}"
    )
    print(
        f"Best localization: {best_loc['feature_set']} / "
        f"{best_loc['model']} | "
        f"Top-1={best_loc['top1_accuracy']:.3f} | "
        f"Stealth Top-1={best_loc['stealth_top1_accuracy']:.3f}"
    )

    prior_simple = detection_df[
        (detection_df["feature_set"] == "prior_only")
        & (detection_df["model"] == "InterpretableScore")
    ].iloc[0]

    topo_candidates = detection_df[
        detection_df["feature_set"] == "topology_fusion"
    ]

    topo_best = topo_candidates.iloc[0]

    print(
        "\nTopology gain over simple max-prior score:\n"
        f"  Balanced accuracy : "
        f"{topo_best['balanced_accuracy'] - prior_simple['balanced_accuracy']:+.3f}\n"
        f"  Stealth recall    : "
        f"{topo_best['stealth_recall'] - prior_simple['stealth_recall']:+.3f}\n"
        f"  Physical FPR      : "
        f"{topo_best['physical_fpr'] - prior_simple['physical_fpr']:+.3f}"
    )

    print(
        "\nINTERPRETATION RULE:\n"
        "- Residual-only is the conventional information baseline.\n"
        "- Prior-only tests whether the separate protected operating prior "
        "already restores visibility.\n"
        "- Topology-fusion is meaningful only if it improves cyber-vs-physical "
        "separation and/or target localization beyond prior-only.\n"
        "- A near-perfect result is NOT yet a final claim: Phase 2Q attacks "
        "have large state bias and broad coherent measurement access. The next "
        "stress phase must reduce attack magnitude, restrict compromised "
        "measurement subsets, and increase prior/load-forecast uncertainty.\n"
        "- A GNN is justified only after this ablation shows that relational "
        "topology contains useful information beyond simple local magnitude."
    )

    print("\nTop Random-Forest topology features:")
    print(importance.head(15).to_string(index=False))

    print("\nSaved:")
    print("  data/phase2s_hard_graph_feature_matrix.csv")
    print("  data/phase2s_hard_node_relational_feature_matrix.csv")
    print("  results/phase2s_hard_cyber_detection_metrics.csv")
    print("  results/phase2s_hard_attack_localization_metrics.csv")
    print("  results/phase2s_hard_topology_rf_feature_importance.csv")
    print("  figures/51_topology_detection_tradeoff_hard.png")
    print("  figures/52_topology_localization_top1_hard.png")
    print("  figures/53_residual_vs_prior_graph_map_hard.png")
    print("  figures/54_topology_fusion_feature_importance_hard.png")


if __name__ == "__main__":
    main()
