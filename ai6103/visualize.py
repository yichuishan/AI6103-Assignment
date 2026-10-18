"""Publication figures: theoretical distributions and measured training curves."""
import csv
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "ai6103-matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "ai6103-cache"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from scipy.special import betaln

from .data import load_json, save_json

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 200, "pdf.fonttype": 42})


def save_figure(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(Path(str(path) + "." + ext), bbox_inches="tight")
    plt.close(fig)


def theoretical_figures(output):
    output = Path(output)
    x = np.linspace(0.0001, 0.9999, 10000)
    density = np.exp(-betaln(0.2, 0.2) - 0.8*np.log(x) - 0.8*np.log1p(-x))
    fig, ax = plt.subplots(figsize=(5.4, 2.7), layout="constrained")
    ax.plot(x, density, color="#126E82", linewidth=1.8)
    ax.set(xlabel=r"Mixing coefficient $\lambda$", ylabel="Probability density (log scale)",
           title=r"$\mathrm{Beta}(0.2,0.2)$: most mass near the endpoints", yscale="log", xlim=(0, 1))
    ax.text(.5, .84, "Density tends to infinity at 0 and 1", transform=ax.transAxes,
            ha="center", fontsize=8)
    ax.grid(alpha=.15)
    save_figure(fig, output / "beta_alpha_0.2")
    t = np.arange(301)
    fig, ax = plt.subplots(figsize=(5.4, 2.7), layout="constrained")
    ax.plot(t, np.ones_like(t), label="Constant", color="#C45B28")
    ax.plot(t, .5*(1+np.cos(np.pi*t/300)), label="Cosine", color="#126E82")
    ax.set(xlabel="Completed epochs", ylabel=r"$\eta_t/\eta_0$", xlim=(0,300), ylim=(-.03,1.08))
    ax.legend(frameon=False)
    ax.grid(alpha=.15)
    save_figure(fig, output / "cosine_schedule")


def read_history(path):
    with Path(path).open(newline="") as f:
        return [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)]


def curves(root, output):
    root, output = Path(root), Path(output)
    completed = {}
    for p in sorted(root.glob("*/summary.json")):
        summary = load_json(p)
        if summary["status"] == "complete":
            completed[summary["config"]["experiment"]["name"]] = summary
    smoke = any(r["config"]["smoke"] for r in completed.values())
    if smoke and output.resolve() == Path("artifacts/figures").resolve():
        raise ValueError("Smoke figures need a separate output, e.g. artifacts/verification/smoke_figures")
    theoretical_figures(output)
    save_json(output / "provenance.json", {"smoke": smoke, "experiments": list(completed),
              "data_fingerprints": sorted({r["config"]["data_fingerprint"] for r in completed.values()})})
    groups = {"learning_rate": ["lr_0.5", "lr_0.005", "lr_0.001"],
              "schedule": ["schedule_constant", "schedule_cosine"],
              "weight_decay": ["schedule_cosine", "decay_0.0005", "decay_0.0001"]}
    if (root / "mixup_baseline.json").exists():
        groups["mixup"] = [load_json(root / "mixup_baseline.json")["name"], "mixup"]
    for group, names in groups.items():
        available = [n for n in names if n in completed]
        if not available:
            continue
        fig, axes = plt.subplots(2, len(available), figsize=(9.0, 4.1),
                                 squeeze=False, layout="constrained")
        for col, name in enumerate(available):
            h = read_history(root / name / "history.csv")
            epochs = [r["epoch"] for r in h]
            for row, metric in enumerate(("loss", "accuracy")):
                ax = axes[row, col]
                for split, color in (("train", "#126E82"), ("val", "#C45B28")):
                    values = [r[f"{split}_{metric}"] * (100 if metric=="accuracy" else 1) for r in h]
                    ax.plot(epochs, values, label="Training" if split=="train" else "Validation",
                            color=color, linestyle="-" if split=="train" else "--")
                ax.set(xlabel="Epoch", ylabel="Cross-entropy" if metric=="loss" else "Top-1 accuracy (%)")
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                if metric == "accuracy":
                    ax.set_ylim(0, 100)
                if row == 0:
                    ax.set_title(name.replace("_", " "))
                    ax.legend(frameon=False, fontsize=7)
                ax.grid(alpha=.15)
        if any(completed[n]["config"]["smoke"] for n in available):
            fig.suptitle("SYNTHETIC SMOKE TEST — NOT ASSIGNMENT RESULTS", fontsize=10, color="#A52A2A")
        save_figure(fig, output / group)
        # A legible single-column overlay for the six-page report. Retain the
        # larger per-condition panels above as standalone investigation figures.
        compact,compact_axes=plt.subplots(2,1,figsize=(3.4,3.5),layout="constrained")
        palette=["#126E82","#C45B28","#5B8C36"]
        all_losses=[]
        for i,name in enumerate(available):
            h=read_history(root/name/"history.csv")
            epochs=[r["epoch"] for r in h]
            label=name.replace("schedule_","").replace("decay_","wd=").replace("lr_","lr=")
            for metric,ax in zip(("loss","accuracy"),compact_axes):
                for split,style in (("train","-"),("val","--")):
                    values=[r[f"{split}_{metric}"]*(100 if metric=="accuracy" else 1) for r in h]
                    ax.plot(epochs,values,color=palette[i%len(palette)],linestyle=style,
                            label=label if split=="train" else None,linewidth=1.1)
                    if metric=="loss": all_losses.extend(values)
                ax.set(xlabel="Epoch",ylabel="CE" if metric=="loss" else "Accuracy (%)")
                ax.xaxis.set_major_locator(MaxNLocator(integer=True,nbins=4))
                ax.grid(alpha=.15)
        positive=[v for v in all_losses if v>0]
        if positive and max(positive)/min(positive)>100:
            compact_axes[0].set_yscale("log")
            compact_axes[0].set_ylabel("CE (log scale)")
        compact_axes[0].legend(frameon=False,fontsize=9,ncol=2,loc="upper center")
        compact_axes[1].set_ylim(0,100)
        if smoke:
            compact.suptitle("SYNTHETIC VERIFICATION",color="#A52A2A",fontsize=9)
        save_figure(compact,output/(group+"_report"))
    return completed
