from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main():
    data1_path = Path("../fem1d.first_order/result.csv")
    data2_path = Path("../fem1d.second_order/result.csv")
    df1 = pd.read_csv(data1_path, header=0, index_col=0)
    df2 = pd.read_csv(data2_path, header=0, index_col=0)

    problems = sorted(set(df1["Problem"]))
    conditions = sorted(set(df1["Condition"]))
    for problem in problems:
        df1_ = df1[df1["Problem"] == problem]
        df2_ = df2[df2["Problem"] == problem]
        problem = str(problem).strip()
        fig, ax = plt.subplots()
        for condition in conditions:
            example1 = df1_[df1_["Condition"] == condition]
            example2 = df2_[df2_["Condition"] == condition]
            label = str(condition).strip()
            ax.plot(
                example1["Number of Nodes"],
                example1["Relative Error"],
                linestyle="solid",
                marker="v",
                label="1st-order, " + label,
            )
            ax.plot(
                example2["Number of Nodes"],
                example2["Relative Error"],
                linestyle="dashed",
                marker="^",
                label="2nd-order, " + label,
            )
        ax.tick_params(axis="both", which="both", direction="in")
        ax.set_xlabel("Number of Nodes")
        ax.set_ylabel("Relative Error")
        ax.set_xscale("log")
        if problem == "laplace":
            ax.set_ylim(-(10 ** (-8)), 10 ** (-8))
        else:
            ax.set_yscale("log")
        ax.minorticks_off()
        ax.legend()
        fig.tight_layout()
        fig.savefig(f"{problem}_relative_error.png")


if __name__ == "__main__":
    main()
