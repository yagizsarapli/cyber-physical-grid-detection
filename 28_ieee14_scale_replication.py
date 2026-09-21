from pathlib import Path
import argparse
import importlib.util
import warnings
from collections import deque

import numpy as np
import pandas as pd
import pandapower as pp
import pandapower.networks as pn

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, recall_score, f1_score, roc_auc_score

warnings.filterwarnings("ignore")

# ============================================================
# PHASE 2V -- SCALE REPLICATION ON A STANDARD TEST SYSTEM (IEEE 14-BUS)
# ============================================================
#
# The one limitation left unaddressed in STATUS.md: everything so far
# ran on a custom 5-bus toy microgrid. This asks whether the core
# finding -- topology-aware features help detection/localization over
# a simpler prior-based baseline -- replicates on IEEE 14-bus, the
# most standard, most commonly cited test system in the power-system
# state-estimation/FDIA literature specifically (unlike the CIGRE MV
# benchmark, which was tried first and rejected here: pandapower
# represents its internal switches with auxiliary ppc buses, breaking
# this project's h_ac()'s bus-index assumptions; IEEE 14-bus has none
# of that and the existing WLS/attack machinery works on it unmodified).
#
# Scope, stated honestly: this is a focused pilot on the CENTRAL
# question (does topology-fusion beat prior-only at detection and
# localization at a bigger, standard scale), not a full re-run of
# every phase (no hard-negative/zero-day/latency work here). Features
# are deliberately redesigned to be size-agnostic (global summary
# statistics + argmax-node topology context) rather than the original
# fixed per-bus columns (b0_.../b4_...), since those don't generalize
# past N=5 -- this is a genuine design improvement, not just a copy.
#
# Every WLS/measurement/attack function below is imported UNMODIFIED
# from 21_..._HARD.py -- build_measurement_model/h_ac/measurement_schema
# all operate generically on net.bus.index/net.line.index/model["n_bus"],
# confirmed by inspection and by this script's own smoke test.
# ============================================================

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
DATA.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

RANDOM_STATE = 20260921


def load_module(filename, module_name):
    path = ROOT / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


phase21 = load_module("21_nonlinear_stealth_fdia_bdd_benchmark_HARD.py", "phase21_for_28")


def graph_adjacency(net):
    adj = {int(b): set() for b in net.bus.index}
    for _, ln in net.line.iterrows():
        f, t = int(ln["from_bus"]), int(ln["to_bus"])
        adj[f].add(t)
        adj[t].add(f)
    for _, tr in net.trafo.iterrows():
        f, t = int(tr["hv_bus"]), int(tr["lv_bus"])
        adj[f].add(t)
        adj[t].add(f)
    return adj


def scale_loads(net, factor, rng=None, per_load_noise=0.0):
    base = net.load["p_mw"].copy()
    baseq = net.load["q_mvar"].copy()
    mult = factor
    if rng is not None and per_load_noise > 0:
        mult = factor * (1.0 + rng.normal(0.0, per_load_noise, size=len(net.load)))
        mult = np.clip(mult, 0.5, 2.0)
    net.load["p_mw"] = base * mult
    net.load["q_mvar"] = baseq * mult


def summary_stats(x, prefix):
    x = np.asarray(x, dtype=float)
    x_sorted = np.sort(x)[::-1]
    return {
        f"{prefix}_mean": float(np.mean(x)),
        f"{prefix}_std": float(np.std(x)),
        f"{prefix}_max": float(np.max(x)),
        f"{prefix}_second": float(x_sorted[1]) if len(x) > 1 else float(x_sorted[0]),
        f"{prefix}_peak_to_mean": float(np.max(x) / (np.mean(x) + 1e-9)),
        f"{prefix}_peak_to_second": float(
            x_sorted[0] / (x_sorted[1] + 1e-9)
        ) if len(x) > 1 else 1.0,
    }


def per_node_rows(scenario_id, case_name, is_cyber, target_bus,
                   node_ids, adjacency, norm_res, innov):
    """One row per node per scenario, for training a REAL topology-aware
    localizer (rank every node's probability of being the attack target)
    instead of the earlier argmax-only baseline."""
    rows = []
    for i, b in enumerate(node_ids):
        nbs = sorted(adjacency[b])
        nb_res = np.array([norm_res[node_ids.index(n)] for n in nbs]) if nbs else np.array([norm_res[i]])
        nb_innov = np.array([innov[node_ids.index(n)] for n in nbs]) if nbs else np.array([innov[i]])
        rows.append({
            "scenario_id": scenario_id,
            "case": case_name,
            "is_cyber": is_cyber,
            "node": b,
            "is_target": int(b == target_bus),
            "node_res": float(norm_res[i]),
            "node_innov": float(innov[i]),
            "node_res_nb_mean": float(np.mean(nb_res)),
            "node_res_nb_max": float(np.max(nb_res)),
            "node_innov_nb_mean": float(np.mean(nb_innov)),
            "node_innov_nb_max": float(np.max(nb_innov)),
            "node_res_local_minus_nb": float(norm_res[i] - np.mean(nb_res)),
            "node_innov_local_minus_nb": float(innov[i] - np.mean(nb_innov)),
            "node_n_neighbors": len(nbs),
        })
    return rows


def argmax_node_topology(node_ids, adjacency, values, target_idx):
    """Topology-relational context for whichever node has the highest
    |values| -- generalizes the original's fixed b{bus}_... columns to
    any network size by only describing the ONE most suspicious node."""
    b = node_ids[target_idx]
    nbs = sorted(adjacency[b])
    if not nbs:
        nb_mean = nb_max = float(values[target_idx])
    else:
        nb_vals = np.array([values[node_ids.index(n)] for n in nbs])
        nb_mean = float(np.mean(nb_vals))
        nb_max = float(np.max(nb_vals))
    return {
        "argmax_local": float(values[target_idx]),
        "argmax_neighbor_mean": nb_mean,
        "argmax_neighbor_max": nb_max,
        "argmax_local_minus_neighbor": float(values[target_idx]) - nb_mean,
        "argmax_n_neighbors": len(nbs),
    }


CASES = ["clean_noisy", "physical_load_disturbance",
         "naive_single_sensor_corruption", "nonlinear_model_consistent_fdia"]
CYBER_CASES = {"naive_single_sensor_corruption", "nonlinear_model_consistent_fdia"}


def one_replication(rep, load_buses, node_ids, adjacency, rng, base_net):
    rows = []
    node_rows = []
    load_factor = float(rng.uniform(0.85, 1.15))

    # Prior: independent small forecast error on top of the true scaling.
    prior_net = pn.case14()
    scale_loads(prior_net, load_factor, rng=rng, per_load_noise=0.05)
    pp.runpp(prior_net)
    vm_prior, va_prior = phase21.truth_state(prior_net)

    target_bus = int(rng.choice(load_buses))

    for case_name in CASES:
        net = pn.case14()
        scale_loads(net, load_factor)
        if case_name == "physical_load_disturbance":
            mult = float(rng.uniform(1.25, 1.70))
            idx = net.load.index[net.load["bus"] == target_bus][0]
            net.load.loc[idx, "p_mw"] *= mult
            net.load.loc[idx, "q_mvar"] *= mult

        pp.runpp(net)
        model = phase21.build_measurement_model(net)
        vm_true, va_true = phase21.truth_state(net)
        h_true = phase21.h_ac(model, vm_true, va_true)
        names, stds = phase21.measurement_schema(net)

        noise = rng.normal(0.0, 1.0, size=len(stds))
        z_clean = h_true + stds * noise
        z_used = z_clean

        if case_name == "naive_single_sensor_corruption":
            z_used, _, _ = phase21.apply_naive_attack(z_clean, names, target_bus, rng)
        elif case_name == "nonlinear_model_consistent_fdia":
            z_used, _, _ = phase21.apply_model_consistent_attack(
                z_clean, h_true, model, vm_true, va_true, target_bus, rng
            )

        result = phase21.run_wls(net, model, z_used, stds)
        if result is None:
            continue
        vm_hat, va_hat = result["vm_hat"], result["va_hat"]

        norm_res = np.abs(vm_hat - vm_true) * 100.0  # pu -> %, keeps scale sane
        innov = np.abs(vm_hat - vm_prior) * 100.0

        row = {
            "rep": rep,
            "case": case_name,
            "is_cyber": int(case_name in CYBER_CASES),
            "target_bus": target_bus if case_name != "clean_noisy" else -1,
        }
        row.update(summary_stats(norm_res, "res"))
        row.update(summary_stats(innov, "innov"))

        res_argmax_idx = int(np.argmax(norm_res))
        innov_argmax_idx = int(np.argmax(innov))
        row["res_argmax_bus"] = node_ids[res_argmax_idx]
        row["innov_argmax_bus"] = node_ids[innov_argmax_idx]
        row.update({f"resnode_{k}": v for k, v in
                     argmax_node_topology(node_ids, adjacency, norm_res, res_argmax_idx).items()})
        row.update({f"innovnode_{k}": v for k, v in
                     argmax_node_topology(node_ids, adjacency, innov, innov_argmax_idx).items()})

        rows.append(row)

        scenario_id = f"{rep}_{case_name}"
        true_target = target_bus if case_name in CYBER_CASES else -1
        node_rows.extend(per_node_rows(
            scenario_id, case_name, row["is_cyber"], true_target,
            node_ids, adjacency, norm_res, innov,
        ))

    return rows, node_rows


def feature_sets(df):
    # NOTE (caught in a later audit, see STATUS.md Sec. 2.5/7.5): the
    # single-argmax topology context (argmax_neighbor_mean/max,
    # argmax_local_minus_neighbor, argmax_n_neighbors -- all pulled in
    # here via the "resnode_"/"innovnode_" prefix) is itself relational,
    # not local. Excluding it by the "neighbor" substring it always
    # contains (argmax_n_neighbors included, since "n_neighbors"
    # contains "neighbor") gives a genuinely topology-free
    # residual_only/prior_only, matching the fix already applied to the
    # 5-bus study's feature_columns(). topology_fusion is unchanged
    # (still everything); residual_plus_prior is the new, fair,
    # topology-free union.
    res_all = [c for c in df.columns if c.startswith("res_") or c.startswith("resnode_")]
    innov_all = [c for c in df.columns if c.startswith("innov_") or c.startswith("innovnode_")]
    residual_only = [c for c in res_all if "neighbor" not in c]
    prior_only = [c for c in innov_all if "neighbor" not in c]
    residual_plus_prior = residual_only + prior_only
    topology_fusion = res_all + innov_all
    return {"residual_only": residual_only, "prior_only": prior_only,
            "residual_plus_prior": residual_plus_prior,
            "topology_fusion": topology_fusion}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-rep", type=int, default=150)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    base_net = pn.case14()
    pp.runpp(base_net)
    load_buses = sorted(base_net.load["bus"].unique().tolist())
    node_ids = sorted(int(b) for b in base_net.bus.index)
    adjacency = graph_adjacency(base_net)

    print(f"\n=== PHASE 2V: IEEE 14-BUS SCALE REPLICATION ===")
    print(f"buses={len(node_ids)} load_buses={len(load_buses)} n_rep={args.n_rep}")

    all_rows = []
    all_node_rows = []
    n_fail = 0
    for rep in range(args.n_rep):
        try:
            r, nr = one_replication(rep, load_buses, node_ids, adjacency, rng, base_net)
            all_rows.extend(r)
            all_node_rows.extend(nr)
        except Exception as e:
            n_fail += 1
            continue
        if (rep + 1) % 25 == 0:
            print(f"  ... {rep + 1}/{args.n_rep} replications done", flush=True)

    df = pd.DataFrame(all_rows)
    node_df = pd.DataFrame(all_node_rows)
    print(f"scenarios: {len(df)} (failed replications: {n_fail})")
    print(df["case"].value_counts().to_string())

    # replication-group-safe split
    reps = df["rep"].unique()
    rng.shuffle(reps)
    n_train = int(round(0.7 * len(reps)))
    n_val = int(round(0.15 * len(reps)))
    train_reps = set(reps[:n_train])
    val_reps = set(reps[n_train:n_train + n_val])
    test_reps = set(reps[n_train + n_val:])
    df["split"] = df["rep"].apply(
        lambda r: "train" if r in train_reps else ("validation" if r in val_reps else "test")
    )

    rep_of_scenario = df.set_index(df["rep"].astype(str) + "_" + df["case"])["split"]
    node_df["split"] = node_df["scenario_id"].map(rep_of_scenario)

    DATA.mkdir(exist_ok=True)
    df.to_csv(DATA / "phase2v_ieee14_feature_matrix.csv", index=False)
    node_df.to_csv(DATA / "phase2v_ieee14_node_feature_matrix.csv", index=False)

    feats = feature_sets(df)
    train_mask = df["split"] == "train"
    test_mask = df["split"] == "test"

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE),
        "HistGradientBoosting": HistGradientBoostingClassifier(random_state=RANDOM_STATE),
    }

    det_rows = []
    loc_rows = []
    for fname, cols in feats.items():
        X_train, y_train = df.loc[train_mask, cols], df.loc[train_mask, "is_cyber"]
        X_test, y_test = df.loc[test_mask, cols], df.loc[test_mask, "is_cyber"]
        for mname, model_obj in models.items():
            clf = Pipeline([("impute", SimpleImputer()), ("scale", StandardScaler()),
                             ("model", model_obj)])
            clf.fit(X_train, y_train)
            pred = clf.predict(X_test)
            proba = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else pred
            det_rows.append({
                "feature_set": fname, "model": mname,
                "balanced_accuracy": balanced_accuracy_score(y_test, pred),
                "f1": f1_score(y_test, pred),
                "recall": recall_score(y_test, pred),
                "roc_auc": roc_auc_score(y_test, proba),
            })

    # Real trained localizer: rank every node's P(is_target) per scenario,
    # not just argmax-residual. Trained/evaluated on cyber scenarios only
    # (mirrors the 5-bus study's localization protocol).
    # Same correction as feature_sets() above: node_res_nb_mean/nb_max/
    # local_minus_nb and node_n_neighbors are all relational/structural,
    # not local -- they do not belong in a topology-free baseline.
    node_feat_sets = {
        "residual_only": ["node_res"],
        "prior_only": ["node_innov"],
        "residual_plus_prior": ["node_res", "node_innov"],
        "topology_fusion": ["node_res", "node_res_nb_mean", "node_res_nb_max",
                             "node_res_local_minus_nb",
                             "node_innov", "node_innov_nb_mean", "node_innov_nb_max",
                             "node_innov_local_minus_nb", "node_n_neighbors"],
    }
    cyber_nodes = node_df[node_df["is_cyber"] == 1]
    train_scn = cyber_nodes[cyber_nodes["split"] == "train"]
    test_scn = cyber_nodes[cyber_nodes["split"] == "test"]

    for fname, cols in node_feat_sets.items():
        loc_clf = Pipeline([("impute", SimpleImputer()), ("scale", StandardScaler()),
                             ("model", HistGradientBoostingClassifier(random_state=RANDOM_STATE))])
        loc_clf.fit(train_scn[cols], train_scn["is_target"])

        test_scn = test_scn.copy()
        test_scn[f"score_{fname}"] = loc_clf.predict_proba(test_scn[cols])[:, 1]

        top1_hits, top2_hits, n_scn = 0, 0, 0
        for sid, g in test_scn.groupby("scenario_id"):
            ranked = g.sort_values(f"score_{fname}", ascending=False)
            true_node = g.loc[g["is_target"] == 1, "node"]
            if true_node.empty:
                continue
            true_node = true_node.iloc[0]
            n_scn += 1
            top1_hits += int(ranked.iloc[0]["node"] == true_node)
            top2_hits += int(true_node in ranked.iloc[:2]["node"].values)

        loc_rows.append({
            "feature_set": fname,
            "top1_accuracy": top1_hits / n_scn if n_scn else float("nan"),
            "top2_accuracy": top2_hits / n_scn if n_scn else float("nan"),
            "n_test_scenarios": n_scn,
        })

    det_df = pd.DataFrame(det_rows).sort_values("balanced_accuracy", ascending=False)
    loc_df = pd.DataFrame(loc_rows).sort_values("top1_accuracy", ascending=False)

    print("\n=== DETECTION (test) ===")
    print(det_df.to_string(index=False))
    print("\n=== LOCALIZATION (argmax-node top-1, test, cyber cases only) ===")
    print(loc_df.to_string(index=False))

    det_df.to_csv(RESULTS / "phase2v_ieee14_detection_metrics.csv", index=False)
    loc_df.to_csv(RESULTS / "phase2v_ieee14_localization_metrics.csv", index=False)
    print("\nSaved:\n  results/phase2v_ieee14_detection_metrics.csv"
          "\n  results/phase2v_ieee14_localization_metrics.csv"
          "\n  data/phase2v_ieee14_feature_matrix.csv"
          "\n  data/phase2v_ieee14_node_feature_matrix.csv")


if __name__ == "__main__":
    main()
