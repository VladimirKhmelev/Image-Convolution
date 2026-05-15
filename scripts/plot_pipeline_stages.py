#!/usr/bin/env python3
"""
Генерирует графики разбивки по этапам пайплайна (3с).

read / process / write — суммарное время по всем изображениям в батче.

Использование:
  python3 scripts/plot_pipeline_stages.py <file1.csv> [file2.csv ...]  [--out <dir>]

Получить CSV:
  bash scripts/bench_pipeline.sh
"""

import os
import csv
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLOR_READ    = "#4E79A7"
COLOR_PROCESS = "#F28E2B"
COLOR_WRITE   = "#E15759"


def read_csv(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        lines = f.readlines()
    headers = [h.strip() for h in lines[0].split(",")]
    reader = csv.DictReader(lines[1:], fieldnames=headers, skipinitialspace=True)
    rows = []
    for row in reader:
        rows.append({
            "filter":         row["filter"].strip(),
            "workers":        int(row["workers"]),
            "batch_size":     int(row["batch_size"]),
            "width":          int(row["width"]),
            "height":         int(row["height"]),
            "sum_read_ms":    float(row["sum_read_ms"]),
            "sum_process_ms": float(row["sum_process_ms"]),
            "sum_write_ms":   float(row["sum_write_ms"]),
            "total_ms":       float(row["total_ms"]),
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
            x_pos = list(range(len(worker_counts)))

            read_vals    = [r["sum_read_ms"]    for r in img_rows]
            process_vals = [r["sum_process_ms"] for r in img_rows]
            write_vals   = [r["sum_write_ms"]   for r in img_rows]

            fig, ax = plt.subplots(figsize=(9, 5))

            ax.bar(x_pos, read_vals, label="Чтение",  color=COLOR_READ,    alpha=0.9,
                   edgecolor="white", linewidth=0.5)
            ax.bar(x_pos, process_vals, label="Свёртка", color=COLOR_PROCESS, alpha=0.9,
                   bottom=read_vals, edgecolor="white", linewidth=0.5)
            ax.bar(x_pos, write_vals, label="Запись",  color=COLOR_WRITE,   alpha=0.9,
                   bottom=[r + p for r, p in zip(read_vals, process_vals)],
                   edgecolor="white", linewidth=0.5)

            totals = [r + p + wv for r, p, wv in zip(read_vals, process_vals, write_vals)]
            max_total = max(totals) if totals else 1
            for xi, total in zip(x_pos, totals):
                ax.text(xi, total + max_total * 0.01,
                        f"{total:.0f}", ha="center", va="bottom", fontsize=8, color="#333333")

            ax.set_xticks(x_pos)
            ax.set_xticklabels([str(wc) for wc in worker_counts])
            ax.set_xlabel("Число воркеров")
            ax.set_ylabel("Суммарное время по батчу (мс)")
            ax.set_title(f"{w}×{h} — {flt} — разбивка по этапам")
            ax.legend(fontsize=9)
            ax.grid(axis="y", linestyle="--", alpha=0.4)
            ax.set_axisbelow(True)
            fig.tight_layout()

            path = os.path.join(out_dir, f"{w}x{h}_stages.png")
            fig.savefig(path, dpi=150)
            plt.close(fig)
            print(f"  Сохранён: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csvfiles", nargs="+", help="CSV-файлы из bench_pipeline.sh")
    parser.add_argument("--out", default="plots/pipeline/stages",
                        help="Папка для PNG (default: plots/pipeline/stages/)")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    all_rows: list[dict] = []
    for path in args.csvfiles:
        all_rows.extend(read_csv(path))

    print(f"Загружено строк: {len(all_rows)}")
    print("Генерация графиков разбивки по этапам...")

    plot_per_image(all_rows, args.out)

    print(f"\nГотово. Графики в папке: {args.out}/")


if __name__ == "__main__":
    main()
