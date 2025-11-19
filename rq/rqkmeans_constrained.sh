#!/bin/bash
#
# RQ-KMeans Constrained Training Script
#

# Default parameters
DATASET="Industrial_and_Scientific"
ROOT="../data/Amazon18/$DATASET"
K=256
L=3
MAX_ITER=100
SEED=42

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset) DATASET="$2"; shift 2 ;;
        --root) ROOT="$2"; shift 2 ;;
        --k) K="$2"; shift 2 ;;
        --l) L="$2"; shift 2 ;;
        --max_iter) MAX_ITER="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

echo "Dataset: $DATASET"
echo "K=$K, L=$L"

python rqkmeans_constrained.py \
    --dataset "$DATASET" \
    --root "$ROOT" \
    --k "$K" \
    --l "$L" \
    --max_iter "$MAX_ITER" \
    --seed "$SEED" \
    --verbose
