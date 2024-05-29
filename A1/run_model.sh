#!/bin/bash

if [ "$1" == "train" ]; then
    python3 train.py --data "$2" --save "$3"
elif [ "$1" == "test" ]; then
    python3 inference.py --model "$2" --test_data "$3" > "$4"
elif [ "$1" == "validate" ]; then
    python3 validation.py --model "$2" --test_data "$3" 
else
    echo "Invalid command. Please use 'train' or 'test'."
fi
