#!/bin/bash

# Check if the correct number of arguments are provided
if [ "$#" -lt 2 ]; then
    echo "Usage for training: $0 train <path_to_data> <path_to_save>"
    echo "Usage for testing: $0 test <path_to_data> <path_to_model> <path_to_result>"
    exit 1
fi

# Assign arguments to variables
mode="$1"
path_to_data="$2"

# Run training mode
if [ "$mode" == "train" ]; then
    if [ "$#" -ne 3 ]; then
        echo "Usage: $0 train <path_to_data> <path_to_save>"
        exit 1
    fi
    path_to_save="$3"
    # Check if train.py exists
    if [ ! -f "train.py" ]; then
        echo "Error: train.py does not exist."
        exit 1
    fi
    # Check if data directory exists
    if [ ! -d "$path_to_data" ]; then
        echo "Error: $path_to_data directory does not exist."
        exit 1
    fi
    # Check if save directory exists, if not create it
    mkdir -p "$path_to_save"
    # Run the training script
    echo "Training the model..."
    python3 train.py "$path_to_data" "$path_to_save"
    echo "Model training completed."

# Run testing mode
elif [ "$mode" == "test" ]; then
    if [ "$#" -ne 4 ]; then
        echo "Usage: $0 test <path_to_data> <path_to_model> <path_to_result>"
        exit 1
    fi
    path_to_model="$3"
    path_to_result="$4"
    # Check if test.py exists
    if [ ! -f "test.py" ]; then
        echo "Error: test.py does not exist."
        exit 1
    fi
    # Check if data directory exists
    if [ ! -d "$path_to_data" ]; then
        echo "Error: $path_to_data directory does not exist."
        exit 1
    fi
    # Check if model directory exists
    if [ ! -d "$path_to_model" ]; then
        echo "Error: $path_to_model directory does not exist."
        exit 1
    fi
    # Check if result directory exists, if not create it
    mkdir -p "$path_to_result"
    # Run the testing script
    echo "Testing the model..."
    python3 test.py "$path_to_data" "$path_to_model" "$path_to_result"
    echo "Model testing completed."

else
    echo "Invalid mode. Usage: $0 [train|test] <path_to_data> <path_to_save|path_to_model> <path_to_result>"
    exit 1
fi
