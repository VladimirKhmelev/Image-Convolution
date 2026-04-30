#!/usr/bin/env python3
"""
Генерирует графики масштабируемости (speedup vs. число потоков) по CSV-данным.

Использование:
  python3 scripts/plot_scaling.py <file1.csv> [file2.csv ...]  [--out <dir>]

Получить CSV:
  bash scripts/bench_scaling.sh
"""

import os
import csv
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

PARALLEL_STRATEGIES = [
    ("По пикселям",  "#4C72B0"),
    ("По строкам",   "#55A868"),
    ("По столбцам",  "#C44E52"),
    ("По сетке",     "#8172B2"),
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


def plot_scaling(all_rows: list[dict], out_dir: str):
    filters = sorted({r["filter"] for r in all_rows})
    for flt in filters:
        frows = [r for r in all_rows if r["filter"] == flt]
        images = sorted({(r["width"], r["height"]) for r in frows},
                        key=lambda wh: wh[0] * wh[1])
        thread_counts = sorted({r["threads"] for r in frows})

        for w, h in images:
            img_rows = [r for r in frows if r["width"] == w and r["height"] == h]

            fig, ax = plt.subplots(figsize=(9, 5))

            max_actual_speedup = 1.0
            for prefix, color in PARALLEL_STRATEGIES:
                xs, ys = [], []
                for t in thread_counts:
                    matches = [r for r in img_rows
                               if r["threads"] == t and r["strategy"].startswith(prefix)]
                    if matches:
                        best = min(matches, key=lambda r: r["mean_ms"])
                        xs.append(t)
                        ys.append(best["speedup"])
                        max_actual_speedup = max(max_actual_speedup, best["speedup"])
                if xs:
                    ax.plot(xs, ys, marker="o", label=prefix,
                            color=color, linewidth=2, markersize=6)

            y_max = max_actual_speedup * 1.15
            # Идеальный линейный speedup, обрезанный по y_max
            ideal_y = [min(t, y_max) for t in thread_counts]
            ax.plot(thread_counts, ideal_y, linestyle="--", color="#AAAAAA",
                    linewidth=1.5, label="Идеальный (линейный)")
            ax.set_ylim(0, y_max)

            ax.set_xlabel("Число потоков")
            ax.set_ylabel("Ускорение (speedup)")
            ax.set_title(f"{w}×{h} — {flt}")
            ax.set_xticks(thread_counts)
            ax.legend(fontsize=9)
            ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
            ax.grid(linestyle="--", alpha=0.4)
            ax.set_axisbelow(True)
            fig.tight_layout()

            path = os.path.join(out_dir, f"{w}x{h}_scaling.png")
            fig.savefig(path, dpi=150)
            plt.close(fig)
            print(f"  Сохранён: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csvfiles", nargs="+", help="CSV-файлы из bench_scaling.sh")
    parser.add_argument("--out", default="plots/scaling", help="Папка для PNG (default: plots/scaling/)")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    all_rows: list[dict] = []
    for path in args.csvfiles:
        all_rows.extend(read_csv(path))

    print(f"Загружено строк: {len(all_rows)}")
    print("Генерация графиков масштабируемости...")

    plot_scaling(all_rows, args.out)

    print(f"\nГотово. Графики в папке: {args.out}/")


if __name__ == "__main__":
    main()
