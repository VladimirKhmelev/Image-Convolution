#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

CSV_DIR="benchmark_data/pipeline"
mkdir -p "$CSV_DIR" plots/pipeline/throughput plots/pipeline/stages plots/pipeline/workers

IMAGES_SMALL="benchmark_images/128x128_cheremsha.jpg"
IMAGES_LARGE="benchmark_images/7071x7071_sea.jpg"

BATCH_SIZE=16
FILTER_3X3='"Gaussian Blur 3×3" "Gaussian Blur 3×3" "Gaussian Blur 3×3"'
FILTER_9X9='"Gaussian Blur 9×9"'

THROUGHPUT_CSV=()
STAGES_CSV=()
WORKERS_CSV=()

echo "=== 3а: пропускная способность (без записи на диск) ==="
for IMG in "$IMAGES_SMALL" "$IMAGES_LARGE"; do
    NAME=$(basename "$IMG" | cut -d. -f1)
    CSV="$CSV_DIR/${NAME}_throughput.csv"
    THROUGHPUT_CSV+=("$CSV")
    echo ""
    echo "--- $IMG ---"
    ./gradlew -q run --args="$IMG $FILTER_3X3 --pipeline --benchmark --no-write --batch-size $BATCH_SIZE --csv $CSV"
done

echo ""
echo "=== 3b: параллельная свёртка внутри воркера (без записи на диск) ==="
for IMG in "$IMAGES_SMALL" "$IMAGES_LARGE"; do
    NAME=$(basename "$IMG" | cut -d. -f1)
    CSV="$CSV_DIR/${NAME}_workers.csv"
    WORKERS_CSV+=("$CSV")
    echo ""
    echo "--- $IMG ---"
    ./gradlew -q run --args="$IMG $FILTER_9X9 --pipeline --benchmark --no-write --vary-threads --worker-strategy rows --workers 2 --batch-size $BATCH_SIZE --csv $CSV"
done

echo ""
echo "=== 3с: разбивка по этапам (с записью на диск) ==="
for IMG in "$IMAGES_SMALL" "$IMAGES_LARGE"; do
    NAME=$(basename "$IMG" | cut -d. -f1)
    CSV="$CSV_DIR/${NAME}_stages.csv"
    STAGES_CSV+=("$CSV")
    echo ""
    echo "--- $IMG ---"
    ./gradlew -q run --args="$IMG $FILTER_3X3 --pipeline --benchmark --batch-size $BATCH_SIZE --csv $CSV"
done

echo ""
echo "=== Строю графики ==="
python3 scripts/plot_pipeline_throughput.py "${THROUGHPUT_CSV[@]}" --out plots/pipeline/throughput
python3 scripts/plot_pipeline_workers.py    "${WORKERS_CSV[@]}"    --out plots/pipeline/workers
python3 scripts/plot_pipeline_stages.py     "${STAGES_CSV[@]}"     --out plots/pipeline/stages

echo ""
echo "Готово. Графики в plots/pipeline/"
