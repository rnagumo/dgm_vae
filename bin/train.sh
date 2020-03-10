
# Specify model name
# export MODEL_NAME=beta
export MODEL_NAME=$1

# Dataset path
export DISENTANGLEMENT_LIB_DATA=./data/
export DATASET_NAME=mnist
export DATA_ROOT=./${DISENTANGLEMENT_LIB_DATA}/${DATASET_NAME}/

# Logging path
export OUTPUT_PATH=./logs/
export EVALUATION_NAME=model/${MODEL_NAME}/${DATASET_NAME}

# Config for training
export CONFIG_PATH=./train/config.json

python ./train/train.py --model ${MODEL_NAME} --epochs 1
