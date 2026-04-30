#!/usr/bin/env python3
"""
Строит графики по per-image CSV из bench_seq.sh.

Использование:
  python3 scripts/plot_seq.py benchmark_data/sequential/*.csv --out plots/sequential
"""

import os
import csv
import re
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def read_csv(path: str) -> dict:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"Пустой файл: {path}")
    row = rows[0]
    return {
        "width":   int(row["width"]),
        "height":  int(row["height"]),
        "mean_ms": float(row["mean_ms"]),
        "std_ms":  float(row["std_ms"]),
    }


def plot_time_vs_size(data: list[dict], out_dir: str):
    data = sorted(data, key=lambda d: d["width"] * d["height"])
    labels = [f"{d['width']}×{d['height']}" for d in data]
    means  = [d["mean_ms"] for d in data]
    stds   = [d["std_ms"]  for d in data]
    x = range(len(data))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, means, marker="o", color="#4C72B0", linewidth=2, markersize=6,
            label="Gaussian Blur 3×3")
    ax.fill_between(x,
                    [m - e for m, e in zip(means, stds)],
                    [m + e for m, e in zip(means, stds)],
                    color="#4C72B0", alpha=0.15)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_xlabel("Размер изображения (пиксели)")
    ax.set_ylabel("Среднее время (мс)")
    ax.set_title("Последовательная свёртка: время vs размер изображения")
    ax.legend(title="Фильтр", fontsize=9)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    fig.tight_layout()

    path = os.path.join(out_dir, "time_vs_size.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Сохранён: {path}")


KERNEL = "Gaussian Blur 3×3"


def pct(data: list[float], p: float) -> float:
    s = sorted(data)
    idx = (len(s) - 1) * p / 100
    lo, hi = int(idx), min(int(idx) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (idx - lo)


def write_csv(path: str, w: int, h: int, times: list[float]):
    import statistics
    with open(path, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["filter", "strategy", "threads", "width", "height",
                     "mean_ms", "std_ms", "p50_ms", "p95_ms"])
        wr.writerow([KERNEL, "Последовательный", 1, w, h,
                     round(statistics.mean(times), 4),
                     round(statistics.stdev(times) if len(times) > 1 else 0, 4),
                     round(pct(times, 50), 4),
                     round(pct(times, 95), 4)])
    print(f"  CSV → {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--write-csv", metavar="PATH",
                        help="Записать summary CSV: PATH W H t1 t2 ...")
    parser.add_argument("csvfiles", nargs="*", help="Per-image CSV файлы для графика")
    parser.add_argument("--out", default="plots/sequential", help="Папка для PNG")
    args = parser.parse_args()

    if args.write_csv:
        # argv after --write-csv: PATH W H t1 t2 ...
        remaining = args.csvfiles  # argparse puts positionals here
        w, h = int(remaining[0]), int(remaining[1])
        times = list(map(float, remaining[2:]))
        write_csv(args.write_csv, w, h, times)
        return

    os.makedirs(args.out, exist_ok=True)
    data = [read_csv(p) for p in args.csvfiles]
    print(f"Загружено файлов: {len(data)}")
    plot_time_vs_size(data, args.out)


if __name__ == "__main__":
    main()
