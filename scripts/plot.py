#!/usr/bin/env python3
"""
Генерирует графики по CSV-данным из benchmark-режима.

Использование:
  python3 scripts/plot.py <file1.csv> [file2.csv ...]  [--out <dir>]

Получить CSV:
  ./gradlew run --args='photo.png "Gaussian Blur 3x3" --benchmark --csv results.csv'
Или через bench_gaussian3.sh, который прогоняет все тестовые изображения сразу.
"""

import sys
import os
import csv
import argparse
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")          # без дисплея
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── цветовая схема ──────────────────────────────────────────────────────────
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

def short_strategy(s: str) -> str:
    """Убирает размер сетки из названия для легенды."""
    return s.split("(")[0].strip()

# ── чтение CSV ───────────────────────────────────────────────────────────────

def read_csv(path: str) -> list[dict]:
    # CSV uses comma as decimal separator, so each float column is split into two fields.
    # Fixed (non-float) columns: filter, strategy, threads, width, height (5 total).
    N_FIXED = 5
    FLOAT_COLS = ["mean_ms", "std_ms", "p50_ms", "p95_ms", "speedup"]

    rows = []
    with open(path, encoding="utf-8") as f:
        lines = [l.rstrip("\n") for l in f if l.strip()]

    for line in lines[1:]:
        parts = line.split(",")
        fixed = parts[:N_FIXED]
        float_parts = parts[N_FIXED:]
        floats = [
            float(f"{float_parts[i * 2]}.{float_parts[i * 2 + 1]}")
            for i in range(len(FLOAT_COLS))
        ]
        row = dict(zip(["filter", "strategy", "threads", "width", "height"], fixed))
        row.update(dict(zip(FLOAT_COLS, floats)))
        rows.append({
            "filter":   row["filter"],
            "strategy": row["strategy"],
            "threads":  int(row["threads"]),
            "width":    int(row["width"]),
            "height":   int(row["height"]),
            "mean_ms":  row["mean_ms"],
            "std_ms":   row["std_ms"],
            "p50_ms":   row["p50_ms"],
            "p95_ms":   row["p95_ms"],
            "speedup":  row["speedup"],
        })
    return rows

# ── График 1: сравнение стратегий (bar chart) ────────────────────────────────
#
#   X: стратегии  |  Y: среднее время (мс)  |  error bars: ±std
#   Показывает относительную скорость всех стратегий на одном изображении.
#   На этом же графике видны кэш-эффекты: BY_COLUMN обычно медленнее BY_ROW.

def plot_strategy_comparison(rows: list[dict], out_dir: str):
    # группируем по фильтру, берём первый найденный thread count (макс)
    filters = sorted({r["filter"] for r in rows})
    for flt in filters:
        frows = [r for r in rows if r["filter"] == flt]
        max_t = max(r["threads"] for r in frows)
        frows = [r for r in frows if r["threads"] == max_t]
        if not frows:
            continue

        # сортируем: seq первый, остальные по времени
        frows_sorted = sorted(frows, key=lambda r: (r["strategy"] != "Последовательный", r["mean_ms"]))

        labels  = [r["strategy"] for r in frows_sorted]
        means   = [r["mean_ms"]  for r in frows_sorted]
        stds    = [r["std_ms"]   for r in frows_sorted]
        colors  = [color_for(r["strategy"]) for r in frows_sorted]
        speedups = [r["speedup"] for r in frows_sorted]

        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(labels))
        bars = ax.bar(x, means, yerr=stds, capsize=4,
                      color=colors, edgecolor="white", linewidth=0.8, alpha=0.9)

        # подписи speedup над столбцами
        for i, (bar, spd) in enumerate(zip(bars, speedups)):
            lbl = f"×{spd:.2f}" if spd != 1.0 else "baseline"
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + stds[i] + max(means) * 0.01,
                    lbl, ha="center", va="bottom", fontsize=9, color="#333333")

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=10)
        ax.set_ylabel("Среднее время (мс)")
        ax.set_title(f"Сравнение стратегий — {flt}  [{frows[0]['width']}×{frows[0]['height']}, {max_t} потоков]")
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)
        fig.tight_layout()

        safe_name = flt.replace(" ", "_").replace("/", "-").replace("→", "to")
        path = os.path.join(out_dir, f"01_comparison_{safe_name}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Сохранён: {path}")

# ── График 2: масштабирование speedup ────────────────────────────────────────
#
#   X: число потоков  |  Y: ускорение относительно sequential
#   Линии: каждая стратегия.  Пунктир: идеальное линейное масштабирование.
#   Показывает закон Амдала в действии: BY_COLUMN хуже масштабируется,
#   BY_ROW и BY_GRID(горизонт.) — лучше.

def plot_speedup_scaling(all_rows: list[dict], out_dir: str):
    filters = sorted({r["filter"] for r in all_rows})
    for flt in filters:
        frows = [r for r in all_rows if r["filter"] == flt]
        thread_counts = sorted({r["threads"] for r in frows})
        if len(thread_counts) < 2:
            continue  # нет смысла строить одну точку

        # стратегии (кроме sequential, он — baseline)
        strategies = sorted({r["strategy"] for r in frows if r["strategy"] != "Последовательный"})

        fig, ax = plt.subplots(figsize=(8, 5))

        for strat in strategies:
            pts = [(r["threads"], r["speedup"]) for r in frows if r["strategy"] == strat]
            pts.sort()
            ts, spds = zip(*pts) if pts else ([], [])
            ax.plot(ts, spds, marker="o", label=short_strategy(strat),
                    color=color_for(strat), linewidth=2, markersize=6)

        # идеальное масштабирование
        ax.plot(thread_counts, thread_counts,
                linestyle="--", color="#AAAAAA", linewidth=1.2, label="Идеальное")

        ax.set_xlabel("Число потоков")
        ax.set_ylabel("Ускорение (×)")
        ax.set_xticks(thread_counts)
        ax.set_title(f"Масштабирование speedup — {flt}  [{frows[0]['width']}×{frows[0]['height']}]")
        ax.legend(fontsize=9)
        ax.grid(linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)
        fig.tight_layout()

        safe_name = flt.replace(" ", "_").replace("/", "-").replace("→", "to")
        path = os.path.join(out_dir, f"02_speedup_{safe_name}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Сохранён: {path}")

# ── График 3: кэш-эффект (BY_ROW vs BY_COLUMN) ──────────────────────────────
#
#   X: размер изображения (N, где картинка N×N)  |  Y: время (мс)
#   Нужно запустить benchmark на картинках разного размера.
#   Данные берутся из CSV, где width == height (квадратные картинки).
#
#   Если в одном CSV только один размер — график не строится, нужен sweep.

def plot_cache_effect(all_rows: list[dict], out_dir: str):
    # берём только квадратные картинки с несколькими размерами
    square_rows = [r for r in all_rows if r["width"] == r["height"]]
    sizes = sorted({r["width"] for r in square_rows})
    if len(sizes) < 2:
        print("  Пропуск графика кэш-эффекта: данные только одного размера.")
        print("  Запусти scripts/sweep.sh --sizes для сбора данных по размерам.")
        return

    filters = sorted({r["filter"] for r in square_rows})
    for flt in filters:
        frows = [r for r in square_rows if r["filter"] == flt]
        max_t = max(r["threads"] for r in frows)
        frows = [r for r in frows if r["threads"] == max_t]

        focus = ["Последовательный", "По строкам", "По столбцам", "По сетке"]
        strategies = [s for s in focus
                      if any(r["strategy"].startswith(s) for r in frows)]

        fig, ax = plt.subplots(figsize=(8, 5))
        for strat in strategies:
            pts = [(r["width"], r["mean_ms"])
                   for r in frows if r["strategy"].startswith(strat)]
            pts.sort()
            if not pts:
                continue
            xs, ys = zip(*pts)
            ax.plot(xs, ys, marker="o", label=strat,
                    color=color_for(strat), linewidth=2, markersize=6)

        ax.set_xlabel("Размер изображения (N, картинка N×N пикс)")
        ax.set_ylabel("Среднее время (мс)")
        ax.set_title(f"Кэш-эффект: BY_ROW vs BY_COLUMN — {flt}  [{max_t} потоков]")
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{int(v)}²"))
        ax.legend(fontsize=9)
        ax.grid(linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)
        fig.tight_layout()

        safe_name = flt.replace(" ", "_").replace("/", "-").replace("→", "to")
        path = os.path.join(out_dir, f"03_cache_{safe_name}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Сохранён: {path}")

# ── График 4: время по стратегиям, линии = изображения ──────────────────────
#
#   X: стратегии  |  Y: среднее время (мс)  |  линии: одна на картинку
#   По сетке — берётся лучший вариант (мин. mean_ms).

STRATEGY_ORDER = [
    ("Последовательный", "Последовательный"),
    ("GPU",              "GPU"),
    ("По пикселям",      "По пикселям"),
    ("По строкам",       "По строкам"),
    ("По столбцам",      "По столбцам"),
    ("По сетке",         "По сетке"),
]

IMAGE_COLORS = ["#E15759", "#4E79A7", "#F28E2B", "#76B7B2", "#59A14F"]


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
                if matches:
                    y_vals.append(min(r["mean_ms"] for r in matches))
                else:
                    y_vals.append(None)

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


# ── График 5: по одному графику на каждое изображение ───────────────────────
#
#   X: стратегии  |  Y: среднее время (мс) + error bars  |  один график = одна картинка

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

        for idx, (w, h) in enumerate(images):
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


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csvfiles", nargs="+", help="CSV-файлы из --benchmark --csv")
    parser.add_argument("--out", default="plots", help="Папка для PNG (default: plots/)")
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
