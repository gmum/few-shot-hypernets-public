

BASE_PATH = $1
export ARGS_PATH = "$BASE_PATH/args.json"
export MODEL_PATH = "$BASE_PATH/best_model.tar"

source activate few-shot-learning

python hypershot_uncertainty.py $(python parse_args.py)