import itertools

import numpy as np
import pytest

from module.discretization import LineMesh, LineMeshHighOrder


class TestLineMesh:
    def test_init(self):
        n_node, xmin, xmax = 4, -2.0, 1.0
        mesh = LineMesh(n_node, xmin, xmax)
        assert mesh.n_node == n_node
        assert mesh.n_element == (n_node - 1)
        assert mesh.xmin == xmin
        assert mesh.xmax == xmax
        assert mesh.boundary_nodes == [0, n_node - 1]
        assert mesh.conditions == ["dirichlet", "dirichlet"]
        assert mesh.x[mesh.boundary_nodes[0]] == mesh.x.min()
        assert mesh.x[mesh.boundary_nodes[1]] == mesh.x.max()
        np.testing.assert_equal(mesh.x, [-2, -1, 0, 1])
        np.testing.assert_equal(mesh.element_nodes, [[0, 1], [1, 2], [2, 3]])
        np.testing.assert_equal(mesh.unit_normals, [-1, 1])

    @pytest.mark.parametrize("conditions", itertools.product(["dirichlet", "neumann"], repeat=2))
    def test_init_conditions(self, conditions):
        n_node, xmin, xmax, conditions = 4, -2.0, 1.0, list(conditions)
        mesh = LineMesh(n_node, xmin, xmax, conditions)
        assert mesh.conditions == conditions

    @pytest.mark.parametrize("conditions", itertools.product(["dirichlet", "neumann"], repeat=2))
    def test_set_conditions(self, conditions):
        n_node, xmin, xmax, conditions = 4, -2.0, 1.0, list(conditions)
        mesh = LineMesh(n_node, xmin, xmax, conditions)
        mesh.conditions = conditions
        assert mesh.conditions == conditions

    def test_set_conditions_exception(self):
        n_node, xmin, xmax = 4, -2.0, 1.0
        mesh = LineMesh(n_node, xmin, xmax)
        with pytest.raises(ValueError):
            mesh.conditions = ["invalid", "condition"]


class TestLineMeshHighOrder:
    def test_init(self):
        n_node, xmin, xmax = 7, -2.0, 4.0
        mesh = LineMeshHighOrder(n_node, xmin, xmax)
        assert mesh.n_node == n_node
        assert mesh.n_element == 3
        assert mesh.xmin == xmin
        assert mesh.xmax == xmax
        assert mesh.boundary_nodes == [0, n_node - 1]
        assert mesh.conditions == ["dirichlet", "dirichlet"]
        assert mesh.x[mesh.boundary_nodes[0]] == mesh.x.min()
        assert mesh.x[mesh.boundary_nodes[1]] == mesh.x.max()
        np.testing.assert_equal(mesh.x, [-2, -1, 0, 1, 2, 3, 4])
        np.testing.assert_equal(mesh.element_nodes, [[0, 2, 1], [2, 4, 3], [4, 6, 5]])
        np.testing.assert_equal(mesh.unit_normals, [-1, 1])

    @pytest.mark.parametrize("conditions", itertools.product(["dirichlet", "neumann"], repeat=2))
    def test_init_conditions(self, conditions):
        n_node, xmin, xmax, conditions = 7, -2.0, 4.0, list(conditions)
        mesh = LineMeshHighOrder(n_node, xmin, xmax, conditions)
        assert mesh.conditions == conditions

    @pytest.mark.parametrize("conditions", itertools.product(["dirichlet", "neumann"], repeat=2))
    def test_set_conditions(self, conditions):
        n_node, xmin, xmax, conditions = 7, -2.0, 4.0, list(conditions)
        mesh = LineMeshHighOrder(n_node, xmin, xmax, conditions)
        mesh.conditions = conditions
        assert mesh.conditions == conditions

    def test_set_conditions_exception(self):
        n_node, xmin, xmax = 7, -2.0, 4.0
        mesh = LineMeshHighOrder(n_node, xmin, xmax)
        with pytest.raises(ValueError):
            mesh.conditions = ["invalid", "condition"]
