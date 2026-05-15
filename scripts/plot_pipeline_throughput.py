#!/usr/bin/env python3
"""
Генерирует графики пропускной способности пайплайна (3а).

Использование:
  python3 scripts/plot_pipeline_throughput.py <file1.csv> [file2.csv ...]  [--out <dir>]

Получить CSV:
  bash scripts/bench_pipeline.sh
"""

import os
import csv
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def read_csv(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        lines = f.readlines()
    headers = [h.strip() for h in lines[0].split(",")]
    reader = csv.DictReader(lines[1:], fieldnames=headers, skipinitialspace=True)
    rows = []
    for row in reader:
        rows.append({
            "filter":       row["filter"].strip(),
            "workers":      int(row["workers"]),
            "batch_size":   int(row["batch_size"]),
            "width":        int(row["width"]),
            "height":       int(row["height"]),
            "throughput":   float(row["throughput_img_s"]),
            "total_ms":     float(row["total_ms"]),
            "speedup":      float(row["speedup"]),
        })
    return rows


def plot_per_image(all_rows: list[dict], out_dir: str):
    filters = sorted({r["filter"] for r in all_rows})
    for flt in filters:
        frows = [r for r in all_rows if r["filter"] == flt]
        images = sorted({(r["width"], r["height"]) for r in frows},
                        key=lambda wh: wh[0] * wh[1])

        for w, h in images:
            img_rows = sorted(
                [r for r in frows if r["width"] == w and r["height"] == h],
                key=lambda r: r["workers"]
            )
            worker_counts = [r["workers"] for r in img_rows]
            throughputs   = [r["throughput"] for r in img_rows]
            speedups      = [r["speedup"] for r in img_rows]
            x_pos = list(range(len(worker_counts)))

            fig, ax1 = plt.subplots(figsize=(9, 5))
            ax2 = ax1.twinx()

            bars = ax1.bar(x_pos, throughputs, color="#4E79A7", alpha=0.85,
                           edgecolor="white", linewidth=0.8, label="изобр/с")
            for bar, val in zip(bars, throughputs):
                ax1.text(bar.get_x() + bar.get_width() / 2,
                         bar.get_height() + max(throughputs) * 0.01,
                         f"{val:.1f}", ha="center", va="bottom", fontsize=8, color="#333333")

            ax2.plot(x_pos, speedups, marker="o", color="#E15759",
                     linewidth=2, markersize=6, label="ускорение")
            ax2.axhline(1.0, linestyle="--", color="#AAAAAA", linewidth=1)

            ax1.set_xlabel("Число воркеров")
            ax1.set_ylabel("Пропускная способность (изобр/с)", color="#4E79A7")
            ax2.set_ylabel("Ускорение (speedup)", color="#E15759")
            ax1.set_title(f"{w}×{h} — {flt} — масштабируемость пайплайна")
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels([str(wc) for wc in worker_counts])

            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="upper left")

            ax1.grid(axis="y", linestyle="--", alpha=0.4)
            ax1.set_axisbelow(True)
            fig.tight_layout()

            path = os.path.join(out_dir, f"{w}x{h}_throughput.png")
            fig.savefig(path, dpi=150)
            plt.close(fig)
            print(f"  Сохранён: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csvfiles", nargs="+", help="CSV-файлы из bench_pipeline.sh")
    parser.add_argument("--out", default="plots/pipeline/throughput",
                        help="Папка для PNG (default: plots/pipeline/throughput/)")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    all_rows: list[dict] = []
    for path in args.csvfiles:
        all_rows.extend(read_csv(path))

    print(f"Загружено строк: {len(all_rows)}")
    print("Генерация графиков пропускной способности...")

    plot_per_image(all_rows, args.out)

    print(f"\nГотово. Графики в папке: {args.out}/")


if __name__ == "__main__":
    main()
