from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

warnings.filterwarnings("ignore")

# ============================================================
# PHASE 2W -- PAPER FIGURES
# ============================================================
# Renders the four figures PAPER_DRAFT.md's own checklist calls for
# (one per results section, §4-§7), from data already produced this
# session. Colors follow the validated categorical palette from the
# project's dataviz skill (references/palette.md): fixed hue order,
# CVD-checked adjacent pairs, not re-picked per chart.
# ============================================================

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)

# --- Validated palette (references/palette.md), light mode ---
BLUE = "#2a78d6"      # slot 1 -- topology_fusion throughout
ORANGE = "#eb6834"    # slot 2 -- prior_only throughout
AQUA = "#1baf7a"       # slot 3 -- residual_only throughout
GOOD = "#0ca30c"
CRITICAL = "#d03b3b"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "axes.edgecolor": BASELINE,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK_SECONDARY,
    "ytick.color": INK_SECONDARY,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})


def style_axes(ax, ygrid=True):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(BASELINE)
    ax.spines["bottom"].set_color(BASELINE)
    if ygrid:
        ax.yaxis.grid(True, color=GRIDLINE, linewidth=1, zorder=0)
        ax.set_axisbelow(True)
    ax.tick_params(length=0)


def bar_labels(ax, bars, fmt="{:.3f}", dy=0.012, color=INK):
    for b in bars:
        h = b.get_height()
        ax.annotate(fmt.format(h), (b.get_x() + b.get_width() / 2, h + dy),
                    ha="center", va="bottom", fontsize=9.5, color=color)


# ============================================================
# Figure 1 -- §4 headline: multi-seed detection + localization.
# topology_fusion vs. the no-topology residual_plus_prior ablation
# (the fair comparison -- feature_columns() originally let
# topology-relational columns leak into residual_only/prior_only by
# name-substring collision; residual_plus_prior is the corrected,
# genuinely topology-free union of the two). prior_only alone is kept
# for scale: it shows how much of the old "topology" gain was really
# just prior_only missing residual information it already had access to.
# ============================================================
seeds = pd.read_csv(RESULTS / "phase2u_multiseed_replication.csv")

series = [
    ("topology_fusion", "topology_fusion", BLUE),
    ("residual_plus_prior", "residual+prior\n(no topology)", AQUA),
    ("prior_only", "prior_only", ORANGE),
]

fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))

for ax, metric, title in [
    (axes[0], "detection_balacc", "Detection (balanced accuracy)"),
    (axes[1], "localization_top1", "Localization (top-1 accuracy)"),
]:
    means = [seeds[f"{key}_best_{metric}"].mean() for key, _, _ in series]
    stds = [seeds[f"{key}_best_{metric}"].std() for key, _, _ in series]
    x = np.arange(len(series))
    colors = [c for _, _, c in series]
    bars = ax.bar(x, means, yerr=stds, capsize=5, width=0.6,
                   color=colors, zorder=3,
                   error_kw={"ecolor": INK_SECONDARY, "elinewidth": 1.3})
    for i, (key, _, _) in enumerate(series):
        jitter = np.linspace(-0.09, 0.09, len(seeds))
        ax.scatter(np.full(len(seeds), i) + jitter, seeds[f"{key}_best_{metric}"],
                    color=INK, s=14, zorder=4, alpha=0.55)
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl, _ in series], fontsize=9)
    ax.set_ylim(0, 1.12)
    ax.set_title(title, fontsize=11, color=INK, pad=10)
    bar_labels(ax, bars, dy=0.05)
    style_axes(ax)

fig.suptitle("5-bus feature ablation across 16 independent seeds",
             fontsize=11, color=INK, y=1.02)
fig.tight_layout()
fig.savefig(FIGURES / "paper_fig1_multiseed_advantage.png", dpi=200, bbox_inches="tight")
plt.close(fig)


# ============================================================
# Figure 2 -- §5 latency waterfall
# ============================================================
# The final bar is read from the current N=300 end-to-end benchmark
# output instead of being hard-coded, so the figure cannot silently
# drift out of sync with the manuscript after a fresh latency run.
latency_summary = pd.read_csv(RESULTS / "phase2t_latency_final.csv")
final_row = latency_summary[
    latency_summary["stage"] == "TOTAL (t5-t0, nothing excluded)"
].iloc[0]
final_median_ms = float(final_row["median_ms"])

steps = [
    ("Original pipeline\n(Random Forest)", 42.98, INK_MUTED),
    ("+ Warm start", 37.14, CRITICAL),
    ("+ In-place measurement\nupdate", 22.62, ORANGE),
    ("+ Logistic regression", 19.60, ORANGE),
    ("Final tuned pipeline\n(N=300)", final_median_ms, GOOD),
]
labels = [s[0] for s in steps]
values = [s[1] for s in steps]
colors = [s[2] for s in steps]

fig, ax = plt.subplots(figsize=(8.5, 4.6))
x = np.arange(len(steps))
bars = ax.bar(x, values, color=colors, width=0.6, zorder=3)
ax.axhline(20.0, color=INK, linewidth=1.4, linestyle=(0, (4, 3)), zorder=2)
ax.annotate("20 ms budget (1 cycle @ 50 Hz)", xy=(0.6, 20.0),
            xytext=(0.6, 20.0 + 3.2), ha="left", fontsize=9, color=INK)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=8.8)
ax.set_ylabel("Median end-to-end latency (ms)")
bar_labels(ax, bars, fmt="{:.1f} ms", dy=0.6)
style_axes(ax)
ax.set_ylim(0, 50)
fig.suptitle("End-to-end latency after successive pipeline optimizations",
             fontsize=11.5, color=INK)
fig.tight_layout()
fig.savefig(FIGURES / "paper_fig2_latency_waterfall.png", dpi=200, bbox_inches="tight")
plt.close(fig)


# ============================================================
# Figure 3 -- §6 zero-day generalization
# ============================================================
zd = pd.read_csv(RESULTS / "phase2u_held_out_attack_type.csv")
if "feature_set" in zd.columns:
    zd = zd[zd["feature_set"] == "topology_fusion"]  # this figure is topology_fusion only; residual_plus_prior rows are reported in the text
zd["attack_type"] = zd["condition"].str.split(":").str[1]
zd["kind"] = zd["condition"].str.split(":").str[0]
pivot = zd.pivot(index="attack_type", columns="kind", values="recall_on_test")
pivot = pivot.rename(index={
    "naive_single_sensor_corruption": "Naive single-sensor\ncorruption",
    "nonlinear_model_consistent_fdia": "Model-consistent\n(stealth) FDIA",
})

fig, ax = plt.subplots(figsize=(7, 4.4))
x = np.arange(len(pivot))
w = 0.35
bars1 = ax.bar(x - w / 2, pivot["seen_in_training"], width=w, color=BLUE,
                label="Seen in training", zorder=3)
bars2 = ax.bar(x + w / 2, pivot["held_out"], width=w, color=CRITICAL,
                label="Attack type withheld", zorder=3)
ax.set_xticks(x)
ax.set_xticklabels(pivot.index, fontsize=10)
ax.set_ylabel("Recall on that attack type")
ax.set_ylim(0, 1.12)
bar_labels(ax, bars1, dy=0.03)
bar_labels(ax, bars2, dy=0.03)
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=2, fontsize=9.5)
style_axes(ax)
fig.tight_layout()
fig.savefig(FIGURES / "paper_fig3_zeroday_generalization.png", dpi=200, bbox_inches="tight")
plt.close(fig)


# ============================================================
# Figure 4 -- §7 scale comparison, 5-bus vs IEEE 14-bus vs IEEE 30-bus
# ============================================================
# FIX (caught in post-submission audit, STATUS.md Sec. 6): this used to
# read the single-run phase2s_hard_*.csv snapshot, which gets
# overwritten by 26_multi_seed_replication.py's LAST seed (2024) --
# so it silently showed one seed's numbers, not the 4-seed mean Table
# III actually reports. Now reads the same phase2u_... 4-seed file
# Figure 1 above uses, mirroring the IEEE-14/30 blocks below exactly
# (residual_only is tracked there too as of this fix).
det5 = {
    "topology_fusion": seeds["topology_fusion_best_detection_balacc"].mean(),
    "residual_plus_prior": seeds["residual_plus_prior_best_detection_balacc"].mean(),
    "prior_only": seeds["prior_only_best_detection_balacc"].mean(),
    "residual_only": seeds["residual_only_best_detection_balacc"].mean(),
}
loc5 = {
    "topology_fusion": seeds["topology_fusion_best_localization_top1"].mean(),
    "residual_plus_prior": seeds["residual_plus_prior_best_localization_top1"].mean(),
    "prior_only": seeds["prior_only_best_localization_top1"].mean(),
    "residual_only": seeds["residual_only_best_localization_top1"].mean(),
}

# IEEE-14: use the 4-seed mean (phase2w_..., from
# 30_ieee14_multi_seed_replication.py), not the single-run
# phase2v_... file -- that file gets overwritten by the last seed in
# the multi-seed loop, so it no longer reflects any one intentional
# run, and Table III/IV in the paper now report 4-seed means for
# IEEE-14 anyway. Consistent with det5/loc5 above using max() per
# feature set (best model), mirrored here per feature set across seeds.
det14w = pd.read_csv(RESULTS / "phase2w_ieee14_multiseed_replication.csv")
det14_best = {
    "topology_fusion": det14w["topology_fusion_det"].mean(),
    "residual_plus_prior": det14w["residual_plus_prior_det"].mean(),
    "prior_only": det14w["prior_only_det"].mean(),
    "residual_only": det14w["residual_only_det"].mean(),
}
loc14_best = {
    "topology_fusion": det14w["topology_fusion_loc"].mean(),
    "residual_plus_prior": det14w["residual_plus_prior_loc"].mean(),
    "prior_only": det14w["prior_only_loc"].mean(),
    "residual_only": det14w["residual_only_loc"].mean(),
}

# IEEE-30: use the 4-seed mean (phase2y_..., from
# 32_ieee30_multi_seed_replication.py), not the single-run phase2x_...
# file -- same reason as IEEE-14 above (that file gets overwritten by
# the last seed in the multi-seed loop). Post-audit (STATUS.md Sec. 6):
# after fixing the oracle-leaked residual and test-set model selection,
# IEEE-30 detection shows a gap positive in all 4 tested seeds (new to
# the audit, not present in any earlier version; not yet a statistically
# confirmed effect at this seed count -- see paper's Sec. VII caveat);
# localization stays null/sign-unstable, unchanged in character.
det30w = pd.read_csv(RESULTS / "phase2y_ieee30_multiseed_replication.csv")
det30_best = {
    "topology_fusion": det30w["topology_fusion_det"].mean(),
    "residual_plus_prior": det30w["residual_plus_prior_det"].mean(),
    "prior_only": det30w["prior_only_det"].mean(),
    "residual_only": det30w["residual_only_det"].mean(),
}
loc30_best = {
    "topology_fusion": det30w["topology_fusion_loc"].mean(),
    "residual_plus_prior": det30w["residual_plus_prior_loc"].mean(),
    "prior_only": det30w["prior_only_loc"].mean(),
    "residual_only": det30w["residual_only_loc"].mean(),
}

order = ["topology_fusion", "residual_plus_prior", "prior_only", "residual_only"]
colors4 = [BLUE, AQUA, ORANGE, INK_SECONDARY]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))

for ax, d5, d14, d30, title in [
    (axes[0], det5, det14_best, det30_best, "Detection (balanced accuracy)"),
    (axes[1], loc5, loc14_best, loc30_best, "Localization (top-1 accuracy)"),
]:
    x = np.arange(3)
    w = 0.19
    for i, (feat, color) in enumerate(zip(order, colors4)):
        vals = [d5[feat], d14[feat], d30[feat]]
        offset = (i - 1.5) * w
        bars = ax.bar(x + offset, vals, width=w, color=color, zorder=3,
                       label=feat if ax is axes[0] else None)
        bar_labels(ax, bars, dy=0.02, fmt="{:.2f}")
    ax.set_xticks(x)
    ax.set_xticklabels(["5-bus\n(16-seed mean)", "IEEE 14-bus\n(16-seed mean)", "IEEE 30-bus\n(16-seed mean)"])
    ax.set_ylim(0, 1.15)
    ax.set_title(title, fontsize=11, color=INK, pad=10)
    style_axes(ax)

axes[0].legend(frameon=False, loc="upper center", bbox_to_anchor=(1.15, 1.16), ncol=4, fontsize=8.8)
fig.tight_layout()
fig.savefig(FIGURES / "paper_fig4_scale_comparison.png", dpi=200, bbox_inches="tight")
plt.close(fig)

print("Saved:")
for f in ["paper_fig1_multiseed_advantage.png", "paper_fig2_latency_waterfall.png",
          "paper_fig3_zeroday_generalization.png", "paper_fig4_scale_comparison.png"]:
    print(f"  figures/{f}")
