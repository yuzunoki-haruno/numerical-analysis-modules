import numpy as np
import pytest
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu

from module.discretization import LineMesh, LineMeshHighOrder
from module.fem import Fem1d


class TestFem1D:
    @pytest.mark.parametrize("conditions", [["D", "D"], ["D", "N"], ["N", "D"]])
    def test_laplace(self, conditions):
        for n in (10, 100, 200):
            xmin, xmax = -0.5, 1
            mesh = LineMesh(n, xmin, xmax, conditions)
            fem = Fem1d(mesh)

            coef, intercept = 2.0, 1.0
            u = coef * mesh.x + intercept
            g = coef * np.ones_like(mesh.x)
            f = np.zeros_like(mesh.x)
            coefficient = fem.laplacian_matrix
            rhs = fem.term(f)

            fem.implement_dirichlet(coefficient, rhs, u)
            fem.implement_neumann(rhs, g)

            sol = splu(csc_matrix(coefficient)).solve(rhs)
            relative_error = np.max(np.abs(u - sol) / np.abs(u).max())
            assert relative_error < 10 ** (-8)

    @pytest.mark.parametrize("conditions", [["D", "D"], ["D", "N"], ["N", "D"]])
    def test_poisson(self, conditions):
        error_old = 1
        for n in (10, 100, 200):
            xmin, xmax = -0.5, 1
            mesh = LineMesh(n, xmin, xmax, conditions)
            fem = Fem1d(mesh)

            coef = 2.0 * np.pi
            coef_x = coef * mesh.x
            u = np.cos(coef_x)
            g = -np.sin(coef_x) * coef
            f = np.cos(coef_x) * coef**2
            coefficient = fem.laplacian_matrix
            rhs = fem.term(f)

            fem.implement_dirichlet(coefficient, rhs, u)
            fem.implement_neumann(rhs, g)

            sol = splu(csc_matrix(coefficient)).solve(rhs)
            relative_error = np.max(np.abs(u - sol) / np.abs(u).max())
            assert relative_error < error_old
            error_old = relative_error

    @pytest.mark.parametrize("conditions", [["D", "D"], ["D", "N"], ["N", "D"]])
    def test_helmholtz(self, conditions):
        error_old = 1
        for n in (10, 100, 200):
            xmin, xmax = -0.5, 1
            mesh = LineMesh(n, xmin, xmax, conditions)
            fem = Fem1d(mesh)

            coef = 2.0 * np.pi
            coef_x = coef * mesh.x
            u = np.cos(coef_x)
            g = -np.sin(coef_x) * coef
            f = np.zeros_like(mesh.x)
            coefficient = fem.laplacian_matrix
            coefficient -= (coef**2) * fem.term_matrix
            rhs = fem.term(f)

            fem.implement_dirichlet(coefficient, rhs, u)
            fem.implement_neumann(rhs, g)

            sol = splu(csc_matrix(coefficient)).solve(rhs)
            relative_error = np.max(np.abs(u - sol) / np.abs(u).max())
            assert relative_error < error_old
            error_old = relative_error


class TestFem1DHighOrder:
    @pytest.mark.parametrize("conditions", [["D", "D"], ["D", "N"], ["N", "D"]])
    def test_laplace(self, conditions):
        for n in (11, 101, 201):
            xmin, xmax = -0.5, 1
            mesh = LineMeshHighOrder(n, xmin, xmax, conditions)
            fem = Fem1d(mesh)

            coef, intercept = 2.0, 1.0
            u = coef * mesh.x + intercept
            g = coef * np.ones_like(mesh.x)
            f = np.zeros_like(mesh.x)
            coefficient = fem.laplacian_matrix
            rhs = fem.term(f)

            fem.implement_dirichlet(coefficient, rhs, u)
            fem.implement_neumann(rhs, g)

            sol = splu(csc_matrix(coefficient)).solve(rhs)
            relative_error = np.max(np.abs(u - sol) / np.abs(u).max())
            assert relative_error < 10 ** (-8)

    @pytest.mark.parametrize("conditions", [["D", "D"], ["D", "N"], ["N", "D"]])
    def test_poisson(self, conditions):
        error_old = 1
        for n in (11, 101, 201):
            xmin, xmax = -0.5, 1
            mesh = LineMeshHighOrder(n, xmin, xmax, conditions)
            fem = Fem1d(mesh)

            coef = 2.0 * np.pi
            coef_x = coef * mesh.x
            u = np.cos(coef_x)
            g = -np.sin(coef_x) * coef
            f = np.cos(coef_x) * coef**2
            coefficient = fem.laplacian_matrix
            rhs = fem.term(f)

            fem.implement_dirichlet(coefficient, rhs, u)
            fem.implement_neumann(rhs, g)

            sol = splu(csc_matrix(coefficient)).solve(rhs)
            relative_error = np.max(np.abs(u - sol) / np.abs(u).max())
            assert relative_error < error_old
            error_old = relative_error

    @pytest.mark.parametrize("conditions", [["D", "D"], ["D", "N"], ["N", "D"]])
    def test_helmholtz(self, conditions):
        error_old = 1
        for n in (11, 101, 201):
            xmin, xmax = -0.5, 1
            mesh = LineMeshHighOrder(n, xmin, xmax, conditions)
            fem = Fem1d(mesh)

            coef = 2.0 * np.pi
            coef_x = coef * mesh.x
            u = np.cos(coef_x)
            g = -np.sin(coef_x) * coef
            f = np.zeros_like(mesh.x)
            coefficient = fem.laplacian_matrix
            coefficient -= (coef**2) * fem.term_matrix
            rhs = fem.term(f)

            fem.implement_dirichlet(coefficient, rhs, u)
            fem.implement_neumann(rhs, g)

            sol = splu(csc_matrix(coefficient)).solve(rhs)
            relative_error = np.max(np.abs(u - sol) / np.abs(u).max())
            assert relative_error < error_old
            error_old = relative_error
