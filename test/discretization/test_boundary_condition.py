import pytest

from module.discretization import BoundaryCondition


class TestBoundaryCondition:
    def test_from_string_dirichlet(self):
        assert BoundaryCondition.DIRICHLET == "dirichlet"
        assert BoundaryCondition.from_string("d") == "dirichlet"
        assert BoundaryCondition.from_string("D") == "dirichlet"
        assert BoundaryCondition.from_string("dirichlet") == "dirichlet"
        assert BoundaryCondition.from_string("Dirichlet") == "dirichlet"
        assert BoundaryCondition.from_string("DIRICHLET") == "dirichlet"

    def test_from_string_neumann(self):
        assert BoundaryCondition.NEUMANN == "neumann"
        assert BoundaryCondition.from_string("n") == "neumann"
        assert BoundaryCondition.from_string("N") == "neumann"
        assert BoundaryCondition.from_string("neumann") == "neumann"
        assert BoundaryCondition.from_string("Neumann") == "neumann"
        assert BoundaryCondition.from_string("NEUMANN") == "neumann"

    def test_from_string_exception(self):
        with pytest.raises(ValueError):
            BoundaryCondition.from_string("cond")

    def test_from_strings_dirichlet(self):
        conditions = ["d", "D", "dirichlet", "Dirichlet", "DIRICHLET"]
        assert BoundaryCondition.from_strings(conditions) == ["dirichlet"] * len(conditions)

    def test_from_strings_neumann(self):
        conditions = ["n", "N", "neumann", "Neumann", "NEUMANN"]
        assert BoundaryCondition.from_strings(conditions) == ["neumann"] * len(conditions)

    def test_from_strings_exception(self):
        conditions = ["D", "N", "None"]
        with pytest.raises(ValueError):
            BoundaryCondition.from_strings(conditions)

    def test_to_indices(self):
        conditions = ["dirichlet", "NEUMANN", "dirichlet", "NEUMANN"]
        dirichlet_indices = BoundaryCondition.to_indices("Dirichlet", conditions)
        neumann_indices = BoundaryCondition.to_indices("Neumann", conditions)
        assert dirichlet_indices == [0, 2]
        assert neumann_indices == [1, 3]

    def test_to_mask(self):
        conditions = ["dirichlet", "NEUMANN", "dirichlet", "NEUMANN"]
        dirichlet_indices = BoundaryCondition.to_mask("Dirichlet", conditions)
        neumann_indices = BoundaryCondition.to_mask("Neumann", conditions)
        assert dirichlet_indices == [True, False, True, False]
        assert neumann_indices == [False, True, False, True]
