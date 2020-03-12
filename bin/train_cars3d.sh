
# Specify model name
# export MODEL_NAME=beta
export MODEL_NAME=$1

# Other parameters
SEED=${2:-0}

# Dataset path
export DISENTANGLEMENT_LIB_DATA=./data/
export DATASET_NAME=cars3d
export DATA_ROOT=./${DISENTANGLEMENT_LIB_DATA}/${DATASET_NAME}/

# Logging path
export OUTPUT_PATH=./logs/
export EVALUATION_NAME=${DATASET_NAME}_${MODEL_NAME}_${SEED}/

# Config for training
export CONFIG_PATH=./src/config_ch3.json

python ./src/train.py --model ${MODEL_NAME} --epochs 100 --seed ${SEED}
