
"""Original evaluation method."""

import json
import os
import pathlib

import numpy as np
import torch

import dgmvae.metrics as dgm


def main():

    # -------------------------------------------------------------------------
    # 1. Settings
    # -------------------------------------------------------------------------

    # Get environment variable
    base_path = os.getenv("OUTPUT_PATH", "./logs/")
    dataset_name = os.getenv("DATASET_NAME", "cars3d")
    experiment_name = os.getenv("EVALUATION_NAME", "tmp")
    root = os.getenv("DATA_ROOT", "./data/cars/")
    config_path = os.getenv("CONFIG_PATH", "./src/metric_config.json")

    # Path config
    experiment_output_path = pathlib.Path(base_path, experiment_name)
    model_path = experiment_output_path.joinpath("representation")
    result_path = experiment_output_path.joinpath("original_metrics")
    result_path.mkdir(exist_ok=True)

    # Random seed
    torch.manual_seed(0)
    np.random.seed(0)

    # -------------------------------------------------------------------------
    # 2. Evaluate
    # -------------------------------------------------------------------------

    # Load metrics config
    with pathlib.Path(config_path).open() as f:
        config = json.load(f)

    # Set evaluator
    evaluator = dgm.MetricsEvaluator(config)
    evaluator.load_dataset(dataset_name, root)
    evaluator.load_model(model_path.joinpath("pytorch_model.pt"))

    # Metrics name and score value name
    evaluation_metrics = {
        "factor_vae_metric": "eval_accuracy",
        "dci": "disentanglement",
        "sap_score": "sap_score",
        "mig": "discrete_mig",
        "irs": "irs",
    }

    final_scores = {}

    for metric_name, score_name in evaluation_metrics.items():
        print(metric_name)

        # Compute metric
        scores_dict = evaluator.compute_metric(metric_name)

        # Log to final score dict
        final_scores[metric_name] = scores_dict[score_name]

    # Save scores
    with result_path.joinpath("local_scores.json").open("w") as f:
        json.dump(final_scores, f)

    print(final_scores)


if __name__ == "__main__":
    main()
