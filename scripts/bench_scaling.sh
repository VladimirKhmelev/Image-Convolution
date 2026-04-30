#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

CSV_DIR="benchmark_data/scaling"
mkdir -p "$CSV_DIR" plots/scaling

IMAGES=(
    "benchmark_images/128x128_cheremsha.jpg"
    "benchmark_images/512x512_zero.png"
    "benchmark_images/1920x1920_pudge.jpeg"
    "benchmark_images/3000x3000_village.jpg"
    "benchmark_images/7071x7071_sea.jpg"
)

THREAD_COUNTS=(1 2 4 8 12)

ALL_CSV_FILES=()

for IMG in "${IMAGES[@]}"; do
    NAME=$(basename "$IMG" | cut -d. -f1)
    echo ""
    echo "=== $IMG ==="
    for T in "${THREAD_COUNTS[@]}"; do
        CSV="$CSV_DIR/${NAME}_t${T}.csv"
        ALL_CSV_FILES+=("$CSV")
        echo "  потоков: $T"
        ./gradlew -q run --args="$IMG \"Gaussian Blur 3×3\" --benchmark --threads $T --csv $CSV"
    done
done

echo ""
echo "=== Строю графики масштабируемости ==="
python3 scripts/plot_scaling.py "${ALL_CSV_FILES[@]}" --out plots/scaling

echo ""
echo "Готово. Графики в plots/scaling/"
