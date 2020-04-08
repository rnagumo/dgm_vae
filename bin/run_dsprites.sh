
# Specify model name
export MODEL_NAME=${1:-beta}

# Other parameters
SEED=${2:-0}

# Dataset path
export DISENTANGLEMENT_LIB_DATA=./data/
export DATASET_NAME=dsprites_full
export DATA_ROOT=./${DISENTANGLEMENT_LIB_DATA}/dsprites/

# Logging path
export OUTPUT_PATH=./logs/
export EVALUATION_NAME=${DATASET_NAME}_${MODEL_NAME}_${SEED}/

# Config for training
export CONFIG_PATH=./src/config_ch1.json

python ./src/train.py --model ${MODEL_NAME} --steps 300000 --seed ${SEED}
python ./src/local_evaluation.py
