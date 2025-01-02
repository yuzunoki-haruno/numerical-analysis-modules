from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main():
    data_path = Path("data.csv")
    df = pd.read_csv(data_path, header=0, index_col=0)

    problems = sorted(set(df["Problem"]))
    conditions = sorted(set(df["Condition"]))
    for problem in problems:
        df_ = df[df["Problem"] == problem]
        problem = str(problem).strip()
        fig, ax = plt.subplots()
        for condition in conditions:
            example = df_[df_["Condition"] == condition]
            label = str(condition).strip()
            ax.scatter(example["Number of Nodes"], example["Relative Error"], label=label)
        ax.tick_params(axis="both", which="both", direction="in")
        ax.set_xlabel("Number of Nodes")
        ax.set_ylabel("Relative Error")
        ax.set_title(f"{problem.title()} Eq.")
        if problem == "laplace":
            ax.set_ylim(-(10 ** (-8)), 10 ** (-8))
        else:
            ax.set_xscale("log")
            ax.set_yscale("log")
        ax.legend()
        fig.tight_layout()
        fig.savefig(data_path.parent / f"Relative_Error_{problem}.png")


if __name__ == "__main__":
    main()
