import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csc_matrix, lil_matrix
from scipy.sparse.linalg import splu

from module.discretization import LineMesh
from module.fem import Fem1d

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="One dimensional finite element analysis")
    prob = ("poisson", "laplace", "helmholtz")
    parser.add_argument("--eq", type=str, required=True, choices=prob, help="Name of governing equation.")
    parser.add_argument("--xmin", type=float, required=True, help="Minimum of x-axis.")
    parser.add_argument("--xmax", type=float, required=True, help="Maximum of x-axis.")
    parser.add_argument("--n_node", type=int, required=True, help="Number of nodes.")
    parser.add_argument("--condition", type=str, required=True, nargs=2, help="Boundary condition.")
    parser.add_argument("--image_path", type=str, default="", help="Display log information.")
    args = parser.parse_args()

    mesh = LineMesh(args.n_node, args.xmin, args.xmax)
    mesh.conditions = args.condition
    fem = Fem1d(mesh)

    if args.eq.lower() == "poisson":
        coef = 2.0 * np.pi
        coef_x = coef * mesh.x
        u = np.cos(coef_x)
        g = -np.sin(coef_x) * coef
        f = np.cos(coef_x) * coef**2
        coefficient = fem.laplacian_matrix
        rhs = fem.term(f)
    elif args.eq.lower() == "laplace":
        coef = 2.0
        intercept = 1.0
        u = coef * mesh.x + intercept
        g = coef * np.ones_like(mesh.x)
        coefficient = fem.laplacian_matrix
        rhs = np.zeros_like(mesh.x)
    elif args.eq.lower() == "helmholtz":
        coef = 2.0 * np.pi
        coef_x = coef * mesh.x
        u = np.cos(coef_x)
        g = -np.sin(coef_x) * coef
        coefficient = fem.laplacian_matrix
        coefficient -= (coef**2) * lil_matrix(fem.term_matrix)
        rhs = np.zeros_like(mesh.x)

    fem.implement_dirichlet(coefficient, rhs, u)
    fem.implement_neumann(rhs, g)

    sol = splu(csc_matrix(coefficient)).solve(rhs)
    relative_error = np.max(np.abs(u - sol) / np.abs(u).max())

    print("Program:", __file__)
    print("Problem: ", args.eq.lower())
    print("Number of Nodes: ", mesh.n_node)
    print("Condition: ", mesh.conditions)
    print("Relative Error: ", relative_error)

    if args.image_path:
        xmin, xmax = mesh.x.min(), mesh.x.max()
        ymin, ymax = min(sol.min(), u.min()), max(sol.max(), u.max())
        fig, ax = plt.subplots()
        ax.plot(mesh.x, sol, label="numerical", linestyle="solid", color="blue")
        ax.plot(mesh.x, u, label="analytical", linestyle="dashed", color="red")
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel("x")
        ax.set_ylabel("u(x)")
        ax.tick_params(axis="both", direction="in")
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.image_path)
