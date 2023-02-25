#!/bin/bash
export ARGS_PATH = "$1/args.json"
export MODEL_PATH = "$1/best_model.tar"

source activate few-shot-learning

python hypershot_uncertainty.py $(python parse_args.py)