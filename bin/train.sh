
export CONFIG_PATH=./src/config.json
export OUTPUT_PATH=./logs/
export DATA_ROOT=./data/mnist/
export DATASET_NAME=mnist
export MODEL_NAME=beta
export EVALUATION_NAME=${MODEL_NAME}/${DATASET_NAME}
python ./src/train.py --model ${MODEL_NAME} --epochs 1
