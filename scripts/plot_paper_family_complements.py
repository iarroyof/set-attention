#!/usr/bin/env python3
import csv
from pathlib import Path
import matplotlib.pyplot as plt

ROOT = Path("/workspace")
#ROOT = Path.home() / "set-attention"
BUNDLE = ROOT / "out" / "paper_complements_bundle"
TABLES = BUNDLE / "tables"
PLOTS = BUNDLE / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 220,
    "savefig.dpi": 300,
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.18,
    "grid.linewidth": 0.6,
    "lines.linewidth": 2.2,
    "lines.markersize": 6.5,
})

FAMILY_ORDER = [("dense","exact"), ("sparse","local_band"), ("linear","landmark")]
FAMILY_LABEL = {
    ("dense","exact"): "Dense",
    ("sparse","local_band"): "Sparse",
    ("linear","landmark"): "Linear",
}
MARKER = {
    ("dense","exact"): "o",
    ("sparse","local_band"): "s",
    ("linear","landmark"): "^",
}

def load_tsv(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))

def xf(v):
    if v in ("", "NA", None):
        return None
    return float(v)

def family_rows(rows, fam, back):
    out = [r for r in rows if r["model.attention_family"] == fam and r["model.backend"] == back]
    return out

# -------- Boundary figure --------
boundary = load_tsv(TABLES / "paper_boundary_family_complements.tsv")

fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.1), constrained_layout=True)

for fam, back in FAMILY_ORDER:
    rows = family_rows(boundary, fam, back)
    rows = sorted(rows, key=lambda r: xf(r["ausa/router_candidate_count_struct_mean"]))
    x = [xf(r["ausa/router_candidate_count_struct_mean"]) for r in rows]
    y1 = [xf(r["val/ppl"]) for r in rows]
    y2 = [xf(r["ausa/routing_entropy_norm"]) for r in rows]
    y3 = [xf(r["ausa/router_top1_weight"]) for r in rows]
    label = FAMILY_LABEL[(fam, back)]
    mk = MARKER[(fam, back)]

    axes[0].plot(x, y1, marker=mk, label=label)
    axes[1].plot(x, y2, marker=mk, label=label)
    axes[2].plot(x, y3, marker=mk, label=label)

axes[0].set_xlabel("Structural candidate count")
axes[0].set_ylabel("Validation perplexity")
axes[0].set_title("Boundary sweep: quality vs candidate count")

axes[1].set_xlabel("Structural candidate count")
axes[1].set_ylabel("Normalized routing entropy")
axes[1].set_title("Boundary sweep: routing entropy")

axes[2].set_xlabel("Structural candidate count")
axes[2].set_ylabel("Router top-1 weight")
axes[2].set_title("Boundary sweep: concentration")

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.08))
fig.savefig(PLOTS / "fig_boundary_family_3panel.png", bbox_inches="tight")

# -------- Pooltau figure --------
pooltau = load_tsv(TABLES / "paper_pooltau_family_complements.tsv")

fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.1), constrained_layout=True)

for fam, back in FAMILY_ORDER:
    rows = family_rows(pooltau, fam, back)
    rows = sorted(rows, key=lambda r: xf(r["model.pooling.tau"]))
    x = [xf(r["model.pooling.tau"]) for r in rows]
    y1 = [xf(r["val/ppl"]) for r in rows]
    # Prefer effective support if present; fallback to neff ratio
    if all(r["ausa/pooling_effective_support"] not in ("", "NA", None) for r in rows):
        y2 = [xf(r["ausa/pooling_effective_support"]) for r in rows]
        y2_label = "Pooling effective support"
        y2_title = "Pooling sweep: effective support"
    else:
        y2 = [xf(r["ausa/pooling_neff_ratio"]) for r in rows]
        y2_label = "Pooling $n_{\\mathrm{eff}}$ ratio"
        y2_title = "Pooling sweep: effective support proxy"
    y3 = [xf(r["ausa/grad_ratio_total_rho_pa"]) for r in rows]
    label = FAMILY_LABEL[(fam, back)]
    mk = MARKER[(fam, back)]

    axes[0].plot(x, y1, marker=mk, label=label)
    axes[1].plot(x, y2, marker=mk, label=label)
    axes[2].plot(x, y3, marker=mk, label=label)

axes[0].set_xlabel(r"Pooling temperature $\tau_{\mathrm{pool}}$")
axes[0].set_ylabel("Validation perplexity")
axes[0].set_title("Pooling sweep: quality")

axes[1].set_xlabel(r"Pooling temperature $\tau_{\mathrm{pool}}$")
axes[1].set_ylabel(y2_label)
axes[1].set_title(y2_title)

axes[2].set_xlabel(r"Pooling temperature $\tau_{\mathrm{pool}}$")
axes[2].set_ylabel(r"End-to-end transport $\rho_{pa}$")
axes[2].set_title("Pooling sweep: transport")

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.08))
fig.savefig(PLOTS / "fig_pooltau_family_3panel.png", bbox_inches="tight")

print("Wrote:")
print(" -", PLOTS / "fig_boundary_family_3panel.png")
print(" -", PLOTS / "fig_pooltau_family_3panel.png")
