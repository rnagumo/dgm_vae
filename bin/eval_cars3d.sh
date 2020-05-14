
# Run evaluation for cars3d (3-ch image).
# $ bash bin/eval_cars3d.sh <model name> <random seed>

# Specify model name
export MODEL_NAME=${1:-beta}

# Other parameters
SEED=${2:-0}

# Dataset path
export DISENTANGLEMENT_LIB_DATA=./data/
export DATASET_NAME=cars3d
export DATA_ROOT=./${DISENTANGLEMENT_LIB_DATA}/cars/

# Logging path
export OUTPUT_PATH=./logs/
export EVALUATION_NAME=${DATASET_NAME}_${MODEL_NAME}_${SEED}/

# Config for training
export CONFIG_PATH=./src/metric_config.json

python3 ./src/evaluate.py
