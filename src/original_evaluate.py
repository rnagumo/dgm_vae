
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

    # Path config
    experiment_output_path = pathlib.Path(base_path, experiment_name)
    module_path = experiment_output_path.joinpath("representation")
    result_path = experiment_output_path.joinpath("original_metrics")

    if not result_path.exists():
        result_path.mkdir()

    # Random seed
    torch.manual_seed(0)
    np.random.seed(0)

    # -------------------------------------------------------------------------
    # 2. Evaluate
    # -------------------------------------------------------------------------

    evaluator = dgm.MetricsEvaluator()
    evaluator.load_dataset(dataset_name, root)
    evaluator.load_model(module_path.joinpath("pytorch_model.pt"))

    evaluation_metrics = {
        "factor_vae_metric": "eval_accuracy",
        "dci": "disentanglement",
        "sap_score": "SAP_score",
        "mig": "discrete_mig",
        "irs": "IRS",
    }
    final_scores = {}

    for metric_name, score_name in evaluation_metrics.items():
        # Compute metric
        scores_dict = evaluator.compute_metric(metric_name)

        # Log to final score dict
        final_scores[metric_name] = scores_dict[score_name]

        # Mkdir
        save_path = result_path.joinpath(metric_name)
        save_path.mkdir()

        # Save json
        with save_path.joinpath("evaluation_results.json").open() as f:
            json.dump(scores_dict, f)

        print(metric_name, scores_dict)

    # Save scores
    with result_path.joinpath("local_scores.json").open() as f:
        json.dump(final_scores, f)


if __name__ == "__main__":
    main()
