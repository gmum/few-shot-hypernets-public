#!/bin/bash
export ARGSPATH="$1/args.json"
export MODELPATH="$1/best_model.tar"

echo "ARGSPATH=$ARGSPATH"
echo "MODELPATH=$MODELPATH"

echo "MODEL ARGUMENTS:"
echo $(python parse_args.py)

python hypershot_uncertainty.py $(python parse_args.py)