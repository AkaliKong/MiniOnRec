#!/bin/bash


PYTHON_SCRIPT="convert_dataset.py"

INPUT_DIR="data/Amazon18/Industrial_and_Scientific"

OUTPUT_DIR="data/Amazon18"

DATASET_NAME="Industrial_and_Scientific"

# ===========================================

echo "Start converting $DATASET_NAME ..."

python $PYTHON_SCRIPT \
    --dataset_name $DATASET_NAME \
    --data_dir $INPUT_DIR \
    --output_dir $OUTPUT_DIR \
    --category $DATASET_NAME \
    --seed 42

echo "Finished!"
