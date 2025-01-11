from typing import List

import numpy as np
from numpy.typing import NDArray

from .boundary_condition import BoundaryCondition


class DiscretizedRegion1D:
    """離散化された一次元領域"""

    def __init__(self, n_node: int, xmin: float, xmax: float, conditions: List[str] | None = None) -> None:
        """離散化された一次元領域の生成

        Args:
            n_node (int): 節点数
            xmin (float): 一次元領域の下限
            xmax (float): 一次元領域の上限
            conditions (List[str] | None, optional): 一次元領域に課された境界条件. Defaults to None.

        Raises:
            ValueError: 節点数n_nodeが1以下の場合に発生
            ValueError: 上限xmaxが下限xmin以下の場合に発生
        """
        if n_node < 2:
            message = "The number of nodes `n_node` must be an integer greater than 1."
            raise ValueError(message)
        if xmax <= xmin:
            message = "The upper limit `xmax` must be greater than the lower limit `xmin`."
            raise ValueError

        self._x = np.linspace(xmin, xmax, num=n_node)
        self._boundary_nodes = [0, n_node - 1]
        self._conditions = self._set_boundary_conditions(conditions)
        self._unit_normals = np.array([-1.0, 1.0], dtype=float)

    @property
    def n_node(self) -> int:
        """節点数

        Returns:
            int: 節点数
        """
        return int(self._x.shape[0])

    @property
    def xmin(self) -> float:
        """一次元領域の下限

        Returns:
            float: 一次元領域の下限
        """
        return float(self._x.min())

    @property
    def xmax(self) -> float:
        """一次元領域の上限

        Returns:
            float: 一次元領域の上限
        """
        return float(self._x.max())

    @property
    def x(self) -> NDArray:
        """節点のx座標"""
        return self._x

    @property
    def boundary_nodes(self) -> List[int]:
        """境界節点の全体接点番号"""
        return self._boundary_nodes

    @property
    def conditions(self) -> List[str]:
        """境界節点に課された境界条件"""
        return self._conditions

    @conditions.setter
    def conditions(self, conditions: List[str]):
        """境界節点に課された境界条件"""
        self._conditions = self._set_boundary_conditions(conditions)

    @property
    def unit_normals(self) -> NDArray:
        """境界節点における外向き単位法線ベクトル"""
        return self._unit_normals

    def _set_boundary_conditions(self, conditions: List[str] | None) -> List[str]:
        """境界条件の初期化"""
        if conditions is None:
            return BoundaryCondition.from_strings(["dirichlet", "dirichlet"])
        else:
            return BoundaryCondition.from_strings(conditions)
