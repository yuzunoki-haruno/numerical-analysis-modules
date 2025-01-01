from pathlib import Path
from typing import Dict, List

import pandas as pd


def text_to_csv(in_path: Path):
    out_path = in_path.with_suffix(".csv")
    print(f"Input  File: {in_path}")
    print(f"Output File: {out_path}")

    data: Dict[str, List[str]] = dict()
    with open(in_path, mode="r") as f:
        for line in f:
            key, value = line.rstrip().split(":")
            data.setdefault(key, list())
            data[key].append(value)

    df = pd.concat([pd.Series(data[key], name=key) for key in data.keys()], axis=1)
    df.to_csv(out_path, encoding="utf8")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert text file to csv file.")
    parser.add_argument("text_path", type=str, help="Text File Path")
    args = parser.parse_args()

    path = Path(args.text_path)
    text_to_csv(path)
