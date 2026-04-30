#!/usr/bin/env python3
"""
Генерирует графики по CSV-данным из benchmark-режима.

Использование:
  python3 scripts/plot.py <file1.csv> [file2.csv ...]  [--out <dir>]

Получить CSV:
  bash scripts/bench_gaussian3.sh
"""

import os
import csv
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

COLORS = {
    "Последовательный": "#888888",
    "По пикселям":      "#4C72B0",
    "По строкам":       "#55A868",
    "По столбцам":      "#C44E52",
    "По сетке":         "#8172B2",
}

def color_for(strategy: str) -> str:
    for key, c in COLORS.items():
        if strategy.startswith(key):
            return c
    return "#AAAAAA"

STRATEGY_ORDER = [
    ("Последовательный", "Последовательный"),
    ("По пикселям",      "По пикселям"),
    ("По строкам",       "По строкам"),
    ("По столбцам",      "По столбцам"),
    ("По сетке",         "По сетке"),
]

IMAGE_COLORS = ["#E15759", "#4E79A7", "#F28E2B", "#76B7B2", "#59A14F"]


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
            "p50_ms":   float(row["p50_ms"]),
            "p95_ms":   float(row["p95_ms"]),
            "speedup":  float(row["speedup"]),
        })
    return rows


def plot_strategies_per_image(all_rows: list[dict], out_dir: str):
    filters = sorted({r["filter"] for r in all_rows})
    for flt in filters:
        frows = [r for r in all_rows if r["filter"] == flt]
        max_t = max(r["threads"] for r in frows)
        frows = [r for r in frows if r["threads"] == max_t]

        images = sorted({(r["width"], r["height"]) for r in frows},
                        key=lambda wh: wh[0] * wh[1])

        fig, ax = plt.subplots(figsize=(12, 6))
        x_labels = [lbl for _, lbl in STRATEGY_ORDER]
        x_pos = list(range(len(x_labels)))

        for idx, (w, h) in enumerate(images):
            img_rows = [r for r in frows if r["width"] == w and r["height"] == h]
            y_vals = []
            for prefix, _ in STRATEGY_ORDER:
                matches = [r for r in img_rows if r["strategy"].startswith(prefix)]
                y_vals.append(min(r["mean_ms"] for r in matches) if matches else None)

            color = IMAGE_COLORS[idx % len(IMAGE_COLORS)]
            valid_x = [x_pos[i] for i, v in enumerate(y_vals) if v is not None]
            valid_y = [v for v in y_vals if v is not None]
            ax.plot(valid_x, valid_y, marker="o", label=f"{w}×{h}",
                    color=color, linewidth=2, markersize=6)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels, rotation=15, ha="right", fontsize=10)
        ax.set_ylabel("Среднее время (мс)")
        ax.set_title(f"Время по стратегиям — {flt}  [{max_t} потоков]")
        ax.legend(title="Изображение", fontsize=9)
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.grid(linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)
        fig.tight_layout()

        safe = flt.replace(" ", "_").replace("/", "-").replace("→", "to")
        path = os.path.join(out_dir, f"overall_chart_{safe}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Сохранён: {path}")


def plot_individual_images(all_rows: list[dict], out_dir: str):
    filters = sorted({r["filter"] for r in all_rows})
    for flt in filters:
        frows = [r for r in all_rows if r["filter"] == flt]
        max_t = max(r["threads"] for r in frows)
        frows = [r for r in frows if r["threads"] == max_t]

        images = sorted({(r["width"], r["height"]) for r in frows},
                        key=lambda wh: wh[0] * wh[1])

        x_labels = [lbl for _, lbl in STRATEGY_ORDER]
        x_pos = list(range(len(x_labels)))

        for w, h in images:
            img_rows = [r for r in frows if r["width"] == w and r["height"] == h]
            means, stds, colors = [], [], []
            for prefix, _ in STRATEGY_ORDER:
                matches = [r for r in img_rows if r["strategy"].startswith(prefix)]
                if matches:
                    best = min(matches, key=lambda r: r["mean_ms"])
                    means.append(best["mean_ms"])
                    stds.append(best["std_ms"])
                    colors.append(color_for(best["strategy"]))
                else:
                    means.append(0)
                    stds.append(0)
                    colors.append("#AAAAAA")

            fig, ax = plt.subplots(figsize=(10, 5))
            bars = ax.bar(x_pos, means, yerr=stds, capsize=4,
                          color=colors, edgecolor="white", linewidth=0.8, alpha=0.9)

            for bar, mean, std in zip(bars, means, stds):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        mean + std + max(means) * 0.01,
                        f"{mean:.1f}", ha="center", va="bottom", fontsize=9, color="#333333")

            ax.set_xticks(x_pos)
            ax.set_xticklabels(x_labels, rotation=15, ha="right", fontsize=10)
            ax.set_ylabel("Среднее время (мс)")
            ax.set_title(f"{w}×{h} — {flt}  [{max_t} потоков]")
            ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
            ax.grid(axis="y", linestyle="--", alpha=0.4)
            ax.set_axisbelow(True)
            fig.tight_layout()

            path = os.path.join(out_dir, f"{w}x{h}_chart.png")
            fig.savefig(path, dpi=150)
            plt.close(fig)
            print(f"  Сохранён: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csvfiles", nargs="+", help="CSV-файлы из --benchmark --csv")
    parser.add_argument("--out", default="plots/strategies", help="Папка для PNG (default: plots/strategies/)")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    all_rows: list[dict] = []
    for path in args.csvfiles:
        all_rows.extend(read_csv(path))

    print(f"Загружено строк: {len(all_rows)}")
    print("Генерация графиков...")

    plot_strategies_per_image(all_rows, args.out)
    plot_individual_images(all_rows, args.out)

    print(f"\nГотово. Графики в папке: {args.out}/")

if __name__ == "__main__":
    main()
