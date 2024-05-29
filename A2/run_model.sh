#!/bin/bash
MODE=$1

if [ "$MODE" == "test" ]; then
    DATA_FILE="$2"
    PREDICTION_FILE="$3"
    echo "Starting Inference..."
    python3 test.py --test "$DATA_FILE" --pred "$PREDICTION_FILE"
else
    DATA_FILE="$1"
    VAL_FILE="$2"
    echo "Starting Training..."
    python3 train.py --train "$DATA_FILE" --val "$VAL_FILE"
fi
