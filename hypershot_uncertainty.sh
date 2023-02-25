#!/bin/bash
export ARGSPATH="$1/args.json"
export MODELPATH="$1/best_model.tar"

source activate few-shot-learning

python hypershot_uncertainty.py $(python parse_args.py)