"""Charts for the feasibility results (reads the CSV files written by run_feasibility.py).

Usage (repository root, .venv activated):
    python scripts/plot_feasibility.py            # Polish labels (report, notes)
    python scripts/plot_feasibility.py --lang en  # English labels (README)

Writes results/feasibility/{success_rates,data_efficiency,training_curves}[_en].png
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # files only, no window
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

RESULTS_DIR = Path("results/feasibility")
RUNS_DIR = Path("runs/feasibility")

# Validated chart palette (light surface): series colors and ink/chrome colors.
SURFACE, INK, INK_2, MUTED, GRID, BASELINE = (
    "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7",
)  # fmt: skip
BLUE, ORANGE = "#2a78d6", "#eb6834"

TEXT = {
    "pl": {
        "success": "Skuteczność (%)",
        "success_title": "Skuteczność na 100 nowych ustawieniach",
        "success_note": "kropki: 3 seedy treningowe, słupek: średnia",
        "expert": "ekspert\n(skrypt)",
        "mlp_dart": "MLP\ndane DART",
        "mlp_clean": "MLP\ndane czyste",
        "demos": "Liczba demonstracji treningowych",
        "efficiency_title": "Ile demonstracji potrzeba? (MLP, dane DART)",
        "expert_ref": "ekspert",
        "mlp_mean": "MLP (średnia z 3 seedów)",
        "epoch": "Epoka",
        "loss": "Strata (skala log.)",
        "curves_title": "Trening MLP (100 demonstracji DART, seed 0)",
        "train": "trening",
        "val": "walidacja",
        "best": "najlepsza epoka",
    },
    "en": {
        "success": "Success rate (%)",
        "success_title": "Success rate on 100 unseen scenes",
        "success_note": "dots: 3 training seeds, bar: mean",
        "expert": "scripted\nexpert",
        "mlp_dart": "MLP\nDART data",
        "mlp_clean": "MLP\nclean data",
        "demos": "Number of training demonstrations",
        "efficiency_title": "How many demonstrations are needed? (MLP, DART data)",
        "expert_ref": "expert",
        "mlp_mean": "MLP (mean of 3 seeds)",
        "epoch": "Epoch",
        "loss": "Loss (log scale)",
        "curves_title": "MLP training (100 DART demos, seed 0)",
        "train": "train",
        "val": "validation",
        "best": "best epoch",
    },
}


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": ["Segoe UI", "DejaVu Sans"],
            "font.size": 10,
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "axes.edgecolor": BASELINE,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "axes.titlesize": 11.5,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK_2,
            "ytick.labelcolor": INK_2,
            "lines.linewidth": 2,
            "lines.solid_capstyle": "round",
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
        }
    )


def read_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as file:
        return list(csv.DictReader(file))


def rates(models: list[dict], data: str, demos: int) -> list[float]:
    return [
        float(m["success_rate"]) * 100
        for m in models
        if m["data"] == data and m["demos"] == str(demos)
    ]


def plot_success_rates(models: list[dict], t: dict, suffix: str) -> Path:
    expert = float(next(m for m in models if m["name"] == "expert")["success_rate"]) * 100
    groups = [(t["expert"], [expert], MUTED), (t["mlp_dart"], rates(models, "DART", 100), BLUE),
              (t["mlp_clean"], rates(models, "clean", 100), BLUE)]  # fmt: skip
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    for x, (_, values, color) in enumerate(groups):
        mean = float(np.mean(values))
        ax.bar(x, mean, width=0.42, color=color, zorder=2)
        if len(values) > 1:
            jitter = np.linspace(-0.08, 0.08, len(values))
            ax.scatter(
                x + jitter, values, s=22, color=INK_2, edgecolor=SURFACE, linewidth=1.2, zorder=3
            )
        label_y = max(values) + 4  # above the seed dots, so they never overlap
        ax.text(x, label_y, f"{mean:.0f}%", ha="center", va="bottom", color=INK, fontsize=10.5)
    ax.set_xticks(range(len(groups)), [g[0] for g in groups])
    ax.set_ylim(0, 115)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel(t["success"])
    ax.grid(axis="x", visible=False)
    ax.set_title(t["success_title"], loc="left")
    ax.text(0, -0.24, t["success_note"], transform=ax.transAxes, color=MUTED, fontsize=8.5)
    path = RESULTS_DIR / f"success_rates{suffix}.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_data_efficiency(models: list[dict], t: dict, suffix: str) -> Path:
    sizes = sorted({int(m["demos"]) for m in models if m["data"] == "DART"})
    values = [rates(models, "DART", n) for n in sizes]
    means = [float(np.mean(v)) for v in values]
    expert = float(next(m for m in models if m["name"] == "expert")["success_rate"]) * 100
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    ax.axhline(expert, color=MUTED, linewidth=1.2, zorder=1)
    ax.text(sizes[0], expert + 2, t["expert_ref"], color=INK_2, fontsize=9, va="bottom")
    for n, v in zip(sizes, values, strict=True):
        ax.scatter([n] * len(v), v, s=16, color=BLUE, alpha=0.35, linewidth=0, zorder=2)
    ax.plot(sizes, means, color=BLUE, marker="o", markersize=6, markeredgecolor=SURFACE,
            markeredgewidth=1.5, zorder=3)  # fmt: skip
    # Label under the second point, where the line and the dots leave free space.
    ax.text(sizes[1] * 1.08, means[1] - 9, t["mlp_mean"], color=INK_2, fontsize=9, va="top")
    ax.set_xscale("log")
    ax.set_xticks(sizes, [str(n) for n in sizes])
    ax.minorticks_off()
    ax.set_ylim(0, 112)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xlabel(t["demos"])
    ax.set_ylabel(t["success"])
    ax.set_title(t["efficiency_title"], loc="left")
    path = RESULTS_DIR / f"data_efficiency{suffix}.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_training_curves(t: dict, suffix: str) -> Path:
    rows = read_csv(RUNS_DIR / "mlp_DART_n100_seed0" / "metrics.csv")
    epochs = [int(r["epoch"]) for r in rows]
    train = [float(r["train_loss"]) for r in rows]
    val = [float(r["val_loss"]) for r in rows]
    best = int(np.argmin(val))
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    ax.plot(epochs, train, color=BLUE, label=t["train"])
    ax.plot(epochs, val, color=ORANGE, label=t["val"])
    ax.axvline(best, color=MUTED, linewidth=1.0, zorder=1)
    ax.text(best + 2, max(train[0], val[0]), t["best"], color=INK_2, fontsize=8.5, va="top")
    ax.set_yscale("log")
    ticks = [0.03, 0.05, 0.1, 0.2, 0.5, 1.0]
    ax.set_yticks(ticks, [f"{tick:g}" for tick in ticks])  # plain numbers, not 10^-1 notation
    ax.minorticks_off()
    ax.set_xlabel(t["epoch"])
    ax.set_ylabel(t["loss"])
    ax.legend(frameon=False, labelcolor=INK_2)
    ax.set_title(t["curves_title"], loc="left")
    path = RESULTS_DIR / f"training_curves{suffix}.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot the feasibility results.")
    parser.add_argument("--lang", choices=["pl", "en"], default="pl")
    args = parser.parse_args()
    style()
    t = TEXT[args.lang]
    suffix = "" if args.lang == "pl" else "_en"
    models = read_csv(RESULTS_DIR / "models.csv")
    for path in [
        plot_success_rates(models, t, suffix),
        plot_data_efficiency(models, t, suffix),
        plot_training_curves(t, suffix),
    ]:
        print(path)


if __name__ == "__main__":
    main()
