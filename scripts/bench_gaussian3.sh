#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

CSV_DIR="benchmark_data"
mkdir -p "$CSV_DIR" plots

IMAGES=(
    "benchmark_images/128x128_cheremsha.jpg"
    "benchmark_images/512x512_zero.png"
    "benchmark_images/1920x1920_pudge.jpeg"
    "benchmark_images/3000x3000_village.jpg"
    "benchmark_images/7071x7071_sea.jpg"
)

CSV_FILES=()

for IMG in "${IMAGES[@]}"; do
    NAME=$(basename "$IMG" | cut -d. -f1)
    CSV="$CSV_DIR/${NAME}.csv"
    CSV_FILES+=("$CSV")
    echo ""
    echo "=== $IMG ==="
    ./gradlew -q run --args="$IMG 'Gaussian Blur 3x3' --benchmark --csv $CSV"
    echo "CSV → $CSV"
done

echo ""
echo "=== Строю графики ==="
python3 scripts/plot.py "${CSV_FILES[@]}" --out plots

echo ""
echo "Готово. Графики в plots/"
