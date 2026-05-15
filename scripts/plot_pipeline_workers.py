#!/usr/bin/env python3
"""
Генерирует графики сравнения последовательной и параллельной свёртки
внутри воркера (задача 3b).

Использование:
  python3 scripts/plot_pipeline_workers.py <file1.csv> [file2.csv ...]  [--out <dir>]

Получить CSV:
  bash scripts/bench_pipeline.sh
"""

import os
import csv
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLOR_SEQ = "#4E79A7"
COLOR_PAR = "#F28E2B"


def read_csv(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        lines = f.readlines()
    headers = [h.strip() for h in lines[0].split(",")]
    reader = csv.DictReader(lines[1:], fieldnames=headers, skipinitialspace=True)
    rows = []
    for row in reader:
        rows.append({
            "filter":         row["filter"].strip(),
            "worker_mode":    row["worker_mode"].strip(),
            "workers":        int(row["workers"]),
            "worker_threads": int(row["worker_threads"]),
            "width":          int(row["width"]),
            "height":         int(row["height"]),
            "throughput":     float(row["throughput_img_s"]),
            "total_ms":       float(row["total_ms"]),
            "sum_process_ms": float(row["sum_process_ms"]),
            "speedup":        float(row["speedup"]),
        })
    return rows


def plot_per_image(all_rows: list[dict], out_dir: str):
    filters = sorted({r["filter"] for r in all_rows})
    for flt in filters:
        frows = [r for r in all_rows if r["filter"] == flt]
        images = sorted({(r["width"], r["height"]) for r in frows},
                        key=lambda wh: wh[0] * wh[1])

        for w, h in images:
            img_rows = [r for r in frows if r["width"] == w and r["height"] == h]

            seq_rows = [r for r in img_rows if r["worker_mode"] == "seq"]
            par_rows = sorted(
                [r for r in img_rows if r["worker_mode"] != "seq"],
                key=lambda r: r["worker_threads"]
            )

            if not seq_rows or not par_rows:
                continue

            seq_tp = seq_rows[0]["throughput"]

            labels = ["seq"] + [r["worker_mode"] for r in par_rows]
            throughputs = [seq_tp] + [r["throughput"] for r in par_rows]
            speedups    = [tp / seq_tp for tp in throughputs]
            x_pos = list(range(len(labels)))
            colors = [COLOR_SEQ] + [COLOR_PAR] * len(par_rows)

            fig, ax1 = plt.subplots(figsize=(9, 5))
            ax2 = ax1.twinx()

            bars = ax1.bar(x_pos, throughputs, color=colors, alpha=0.85,
                           edgecolor="white", linewidth=0.8)
            for bar, val in zip(bars, throughputs):
                ax1.text(bar.get_x() + bar.get_width() / 2,
                         bar.get_height() + max(throughputs) * 0.01,
                         f"{val:.1f}", ha="center", va="bottom", fontsize=8, color="#333333")

            ax2.plot(x_pos, speedups, marker="o", color="#E15759",
                     linewidth=2, markersize=6, label="ускорение")
            ax2.axhline(1.0, linestyle="--", color="#AAAAAA", linewidth=1)

            workers_count = par_rows[0]["workers"] if par_rows else "?"
            ax1.set_xlabel(f"Режим воркера  ({workers_count} воркера, Gaussian 9×9)")
            ax1.set_ylabel("Пропускная способность (изобр/с)", color="#4E79A7")
            ax2.set_ylabel("Ускорение vs seq", color="#E15759")
            ax1.set_title(f"{w}×{h} — параллельная свёртка внутри воркера")
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels(labels, fontsize=9)

            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor=COLOR_SEQ, alpha=0.85, label="Sequential"),
                Patch(facecolor=COLOR_PAR, alpha=0.85, label="Parallel (rows×N)"),
            ]
            ax1.legend(handles=legend_elements, fontsize=9, loc="upper left")

            ax1.grid(axis="y", linestyle="--", alpha=0.4)
            ax1.set_axisbelow(True)
            fig.tight_layout()

            path = os.path.join(out_dir, f"{w}x{h}_workers.png")
            fig.savefig(path, dpi=150)
            plt.close(fig)
            print(f"  Сохранён: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csvfiles", nargs="+", help="CSV-файлы из bench_pipeline.sh (3b)")
    parser.add_argument("--out", default="plots/pipeline/workers",
                        help="Папка для PNG (default: plots/pipeline/workers/)")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    all_rows: list[dict] = []
    for path in args.csvfiles:
        all_rows.extend(read_csv(path))

    print(f"Загружено строк: {len(all_rows)}")
    print("Генерация графиков параллельной свёртки в воркере...")

    plot_per_image(all_rows, args.out)

    print(f"\nГотово. Графики в папке: {args.out}/")


if __name__ == "__main__":
    main()
