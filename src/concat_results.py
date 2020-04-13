
"""Concats downloaded reults."""

import json
import pathlib

import pandas as pd


def concat_dis_lib(save_dir):
    """Concats results of disentanglement_lib."""

    # Path
    logdir = pathlib.Path("../data/pretrained/")

    # Metrics
    evaluation_metrics = {
        "beta_vae_sklearn": "eval_accuracy",
        "dci": "disentanglement",
        "downstream_task_boosted_trees": "10:mean_test_accuracy",
        "downstream_task_logistic_regression": "10:mean_test_accuracy",
        "factor_vae_metric": "eval_accuracy",
        "mig": "discrete_mig",
        "modularity_explicitness": "modularity_score",
        "sap_score": "SAP_score",
        "unsupervised": "mutual_info_score",
    }

    # Results data frame
    results = {k: [] for k in evaluation_metrics}

    for metric, value_name in evaluation_metrics.items():
        # Glob pattern
        pattern = (f"*/metrics/mean/{metric}/results/json/"
                   + "evaluation_results.json")

        # Read json files
        for path in logdir.glob(pattern):
            with path.open() as f:
                d = json.load(f)

            results[metric].append(d[value_name])

    # Save results
    save_dir = pathlib.Path(save_dir)
    if not save_dir.exists():
        save_dir.mkdir()

    df = pd.DataFrame(results)
    df.to_csv(save_dir.joinpath("dislib_results.csv"), index=None)


if __name__ == "__main__":
    concat_dis_lib("../data/results/")
