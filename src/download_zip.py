
"""Downloads Zip files from google storage.

ref)
https://github.com/google-research/disentanglement_lib

Request library: Raw Response Content
https://requests.readthedocs.io/en/master/user/quickstart/#raw-response-content
"""

import argparse
import pathlib

import requests
import tqdm


def download_data(url, path, chunk_size=128):
    res = requests.get(url, stream=True)
    with open(path, "wb") as fd:
        for chunk in res.iter_content(chunk_size=chunk_size):
            fd.write(chunk)


def main():
    # Keyword args
    parser = argparse.ArgumentParser(description="Download pretrained models")
    parser.add_argument("--exp_num", type=int, default=0,
                        help="Experiment number")
    parser.add_argument("--seed_num", type=int, default=0,
                        help="Number of random seed")
    args = parser.parse_args()

    # Path which downloaded zip files are saved into
    path = pathlib.Path("../data/pretrained/")
    if not path.exists():
        path.mkdir()

    # URL
    url = ("https://storage.googleapis.com/disentanglement_lib/"
           + "unsupervised_study_v1/{}.zip")

    # Download all data
    for n in tqdm.tqdm(range(args.exp_num, args.exp_num + args.seed_num)):
        download_data(url.format(n), path.joinpath(f"{n}.zip"))


if __name__ == "__main__":
    main()
