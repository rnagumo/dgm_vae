
# Specify model name
# export MODEL_NAME=beta
export MODEL_NAME=$1

# Dataset path
export DISENTANGLEMENT_LIB_DATA=./data/
export DATASET_NAME=mnist
export DATA_ROOT=./${DISENTANGLEMENT_LIB_DATA}/${DATASET_NAME}/

# Logging path
export OUTPUT_PATH=./logs/
export SAVE_PATH=${OUTPUT_PATH}/${MODEL_NAME}/
export EVALUATION_NAME=${MODEL_NAME}/model/${DATASET_NAME}

python ./src/local_evaluation.py
