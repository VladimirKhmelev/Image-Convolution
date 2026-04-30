#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

WARMUP=10
RUNS=30
KERNEL="Gaussian Blur 3×3"
CSV_DIR="benchmark_data/sequential"
PLOT_DIR="plots/sequential"

IMAGES=(
    "benchmark_images/128x128_cheremsha.jpg"
    "benchmark_images/512x512_zero.png"
    "benchmark_images/1920x1920_pudge.jpeg"
    "benchmark_images/3000x3000_village.jpg"
    "benchmark_images/7071x7071_sea.jpg"
)

mkdir -p "$CSV_DIR" "$PLOT_DIR"

CSV_FILES=()

for IMG in "${IMAGES[@]}"; do
    NAME=$(basename "$IMG" | cut -d. -f1)
    SIZE=$(echo "$NAME" | grep -oP '^\d+x\d+')
    W=$(echo "$SIZE" | cut -dx -f1)
    H=$(echo "$SIZE" | cut -dx -f2)
    CSV="$CSV_DIR/${NAME}.csv"
    CSV_FILES+=("$CSV")

    echo ""
    echo "=== $NAME ($W×$H) | $KERNEL ==="

    echo -n "  прогрев ($WARMUP)..."
    for ((i=1; i<=WARMUP; i++)); do
        ./gradlew -q run --args="$IMG \"$KERNEL\"" 2>/dev/null
    done
    echo " готово"

    TIMES=()
    for ((i=1; i<=RUNS; i++)); do
        MS=$(./gradlew -q run --args="$IMG \"$KERNEL\"" 2>/dev/null \
             | grep -oP 'Время\s+:\s+\K\d+')
        echo "    run $i: ${MS} мс"
        TIMES+=("$MS")
    done

    python3 scripts/plot_seq.py --write-csv "$CSV" "$W" "$H" "${TIMES[@]}"
done

echo ""
echo "=== Строю графики ==="
python3 scripts/plot_seq.py "${CSV_FILES[@]}" --out "$PLOT_DIR"
echo ""
echo "Готово. Данные в $CSV_DIR/, графики в $PLOT_DIR/"
