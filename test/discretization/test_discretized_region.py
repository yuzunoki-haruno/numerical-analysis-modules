import itertools

import numpy as np
import pytest

from module.discretization import DiscretizedRegion1D


class TestDiscretizedRegion1D:
    def test_init(self):
        n_node, xmin, xmax = 7, -2.0, 1.0
        discretized_region = DiscretizedRegion1D(n_node, xmin, xmax)
        assert discretized_region.n_node == n_node
        assert discretized_region.xmin == xmin
        assert discretized_region.xmax == xmax
        assert discretized_region.boundary_nodes == [0, n_node - 1]
        assert discretized_region.conditions == ["dirichlet", "dirichlet"]
        np.testing.assert_equal(discretized_region.x, [-2, -1.5, -1, -0.5, 0, 0.5, 1])
        np.testing.assert_equal(discretized_region.unit_normals, [-1, 1])

    @pytest.mark.parametrize("conditions", itertools.product(["dirichlet", "neumann"], repeat=2))
    def test_init_conditions(self, conditions):
        n_node, xmin, xmax, conditions = 7, -2.0, 1.0, list(conditions)
        discretized_region = DiscretizedRegion1D(n_node, xmin, xmax, conditions)
        assert discretized_region.conditions == conditions

    @pytest.mark.parametrize("conditions", itertools.product(["dirichlet", "neumann"], repeat=2))
    def test_set_conditions(self, conditions):
        n_node, xmin, xmax, conditions = 7, -2.0, 1.0, list(conditions)
        discretized_region = DiscretizedRegion1D(n_node, xmin, xmax)
        discretized_region.conditions = conditions
        assert discretized_region.conditions == conditions

    def test_set_conditions_exception(self):
        n_node, xmin, xmax = 7, -2.0, 1.0
        discretized_region = DiscretizedRegion1D(n_node, xmin, xmax)
        with pytest.raises(ValueError):
            discretized_region.conditions = ["invalid", "condition"]
