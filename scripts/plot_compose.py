#!/usr/bin/env python3
"""
Сравнительные графики: последовательное применение фильтров vs составное ядро.

Использование:
  python3 scripts/plot_compose.py <file1.csv> [file2.csv ...]  [--out <dir>]
"""

import os
import re
import csv
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

SEQ_COLOR  = "#4C72B0"
COMP_COLOR = "#DD8452"

IMAGE_COLORS = ["#E15759", "#4E79A7", "#F28E2B", "#76B7B2", "#59A14F"]

STRATEGY_ORDER = [
    ("Последовательный", "Последовательный"),
    ("По пикселям",      "По пикселям"),
    ("По строкам",       "По строкам"),
    ("По столбцам",      "По столбцам"),
    ("По сетке",         "По сетке"),
]


def read_csv(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        lines = f.readlines()
    headers = [h.strip() for h in lines[0].split(",")]
    reader = csv.DictReader(lines[1:], fieldnames=headers, skipinitialspace=True)
    rows = []
    for row in reader:
        rows.append({
            "filter":   row["filter"].strip(),
            "strategy": row["strategy"].strip(),
            "threads":  int(row["threads"]),
            "width":    int(row["width"]),
            "height":   int(row["height"]),
            "mean_ms":  float(row["mean_ms"]),
            "std_ms":   float(row["std_ms"]),
            "speedup":  float(row["speedup"]),
        })
    return rows


def split_filters(all_rows: list[dict]) -> tuple[list[dict], list[dict]]:
    seq  = [r for r in all_rows if "(составной" not in r["filter"]]
    comp = [r for r in all_rows if "(составной" in r["filter"]]
    return seq, comp


def best_for_strategy(rows: list[dict], prefix: str) -> tuple[float, float] | tuple[None, None]:
    matches = [r for r in rows if r["strategy"].startswith(prefix)]
    if not matches:
        return None, None
    best = min(matches, key=lambda r: r["mean_ms"])
    return best["mean_ms"], best["std_ms"]


def plot_per_image(all_rows: list[dict], out_dir: str) -> None:
    seq_rows, comp_rows = split_filters(all_rows)

    images = sorted({(r["width"], r["height"]) for r in all_rows},
                    key=lambda wh: wh[0] * wh[1])

    n = len(STRATEGY_ORDER)
    bar_w = 0.35
    x = np.arange(n)
    x_labels = [lbl for _, lbl in STRATEGY_ORDER]

    for w, h in images:
        max_t = max(r["threads"] for r in all_rows if r["width"] == w and r["height"] == h)
        seq_img  = [r for r in seq_rows  if r["width"] == w and r["height"] == h and r["threads"] == max_t]
        comp_img = [r for r in comp_rows if r["width"] == w and r["height"] == h and r["threads"] == max_t]

        comp_size = "?"
        seq_passes = 1
        if comp_img:
            m = re.search(r"\(составной\s+(\d+[×x]\d+)\)", comp_img[0]["filter"])
            if m:
                comp_size = m.group(1)
        if seq_img:
            seq_passes = seq_img[0]["filter"].count("→") + 1

        seq_means, seq_stds, comp_means, comp_stds = [], [], [], []
        for prefix, _ in STRATEGY_ORDER:
            sm, ss = best_for_strategy(seq_img, prefix)
            cm, cs = best_for_strategy(comp_img, prefix)
            seq_means.append(sm or 0); seq_stds.append(ss or 0)
            comp_means.append(cm or 0); comp_stds.append(cs or 0)

        fig, ax = plt.subplots(figsize=(12, 6))

        b1 = ax.bar(x - bar_w / 2, seq_means, bar_w,
                    yerr=seq_stds, capsize=4,
                    color=SEQ_COLOR, alpha=0.85, edgecolor="white", linewidth=0.8,
                    label=f"Последовательное ({seq_passes} прохода)")
        b2 = ax.bar(x + bar_w / 2, comp_means, bar_w,
                    yerr=comp_stds, capsize=4,
                    color=COMP_COLOR, alpha=0.85, edgecolor="white", linewidth=0.8,
                    label=f"Составной {comp_size} (1 проход)")

        max_val = max(max(seq_means), max(comp_means), 1e-9)
        for bar, mean, std in list(zip(b1, seq_means, seq_stds)) + list(zip(b2, comp_means, comp_stds)):
            if mean > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        mean + std + max_val * 0.015,
                        f"{mean:.1f}", ha="center", va="bottom", fontsize=8, color="#333333")

        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=15, ha="right", fontsize=10)
        ax.set_ylabel("Среднее время (мс)")
        ax.set_title(f"{w}×{h} — Последовательное vs Составной ядро  [{max_t} потоков]")
        ax.legend(fontsize=10)
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)
        fig.tight_layout()

        path = os.path.join(out_dir, f"{w}x{h}_compare.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Сохранён: {path}")


def plot_overall(all_rows: list[dict], out_dir: str) -> None:
    seq_rows, comp_rows = split_filters(all_rows)

    images = sorted({(r["width"], r["height"]) for r in all_rows},
                    key=lambda wh: wh[0] * wh[1])

    x_labels = [lbl for _, lbl in STRATEGY_ORDER]
    x_pos = list(range(len(STRATEGY_ORDER)))

    fig, ax = plt.subplots(figsize=(13, 6))

    for idx, (w, h) in enumerate(images):
        max_t = max(r["threads"] for r in all_rows if r["width"] == w and r["height"] == h)
        seq_img  = [r for r in seq_rows  if r["width"] == w and r["height"] == h and r["threads"] == max_t]
        comp_img = [r for r in comp_rows if r["width"] == w and r["height"] == h and r["threads"] == max_t]

        color = IMAGE_COLORS[idx % len(IMAGE_COLORS)]

        seq_vals, comp_vals = [], []
        for prefix, _ in STRATEGY_ORDER:
            sm, _ = best_for_strategy(seq_img, prefix)
            cm, _ = best_for_strategy(comp_img, prefix)
            seq_vals.append(sm)
            comp_vals.append(cm)

        valid_xs = [x_pos[i] for i, v in enumerate(seq_vals) if v is not None]
        valid_ys = [v for v in seq_vals if v is not None]
        ax.plot(valid_xs, valid_ys, marker="o", color=color, linewidth=2, markersize=6,
                linestyle="-", label=f"{w}×{h} (посл.)")

        valid_xc = [x_pos[i] for i, v in enumerate(comp_vals) if v is not None]
        valid_yc = [v for v in comp_vals if v is not None]
        ax.plot(valid_xc, valid_yc, marker="s", color=color, linewidth=2, markersize=6,
                linestyle="--", label=f"{w}×{h} (сост.)")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, rotation=15, ha="right", fontsize=10)
    ax.set_ylabel("Среднее время (мс)")
    ax.set_title("Все изображения — Последовательное (─) vs Составной (- -)")
    ax.legend(title="Размер (тип)", fontsize=8, ncol=2)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    fig.tight_layout()

    path = os.path.join(out_dir, "overall_compare.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Сохранён: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csvfiles", nargs="+", help="CSV-файлы из --benchmark --csv")
    parser.add_argument("--out", default="plots/compose", help="Папка для PNG")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    all_rows: list[dict] = []
    for path in args.csvfiles:
        all_rows.extend(read_csv(path))

    print(f"Загружено строк: {len(all_rows)}")
    print("Генерация графиков...")

    plot_per_image(all_rows, args.out)
    plot_overall(all_rows, args.out)

    print(f"\nГотово. Графики в: {args.out}/")


if __name__ == "__main__":
    main()
