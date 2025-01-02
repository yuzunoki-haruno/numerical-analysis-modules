from typing import List

from .discretized_region import DiscretizedRegion1D


class LineMesh(DiscretizedRegion1D):
    """一次元有限要素（一次要素）"""

    def __init__(self, n_node: int, xmin: float, xmax: float, conditions: List[str] | None = None) -> None:
        """一次元有限要素（一次要素）

        Args:
            n_node (int): 節点数
            xmin (float): 一次元領域の下限
            xmax (float): 一次元領域の上限
            conditions (List[str] | None, optional): 一次元領域に課された境界条件. Defaults to None.
        """
        super().__init__(n_node, xmin, xmax, conditions)
        n_element = n_node - 1
        self._element_nodes = [[i, i + 1] for i in range(n_element)]

    @property
    def n_element(self) -> int:
        """要素数

        Returns:
            int: 要素数
        """
        return len(self._element_nodes)

    @property
    def element_nodes(self) -> List[List[int]]:
        """要素を構成する節点番号のリスト

        Returns:
            List[List[int]]: _description_
        """
        return self._element_nodes

    def __getitem__(self, index: int) -> List[int]:
        """_summary_

        Args:
            index (int): 要素番号

        Returns:
            List[int]: 要素を構成する節点番号
        """
        return self._element_nodes[index]


class LineMeshHighOrder(DiscretizedRegion1D):
    """一次元有限要素（二次要素）"""

    def __init__(self, n_node: int, xmin: float, xmax: float, conditions: List[str] | None = None) -> None:
        """一次元有限要素（二次要素）

        Args:
            n_node (int): 節点数（３以上の奇数）
            xmin (float): 一次元領域の下限
            xmax (float): 一次元領域の上限
            conditions (List[str] | None, optional): 一次元領域に課された境界条件. Defaults to None.
        """
        super().__init__(n_node, xmin, xmax, conditions)
        n_element = n_node // 2
        self._element_nodes = [[2 * i, 2 * (i + 1), 2 * i + 1] for i in range(n_element)]

    @property
    def n_element(self) -> int:
        """要素数

        Returns:
            int: 要素数
        """
        return len(self._element_nodes)

    @property
    def element_nodes(self) -> List[List[int]]:
        """要素を構成する節点番号のリスト

        Returns:
            List[List[int]]: 要素を構成する節点番号のリスト
        """
        return self._element_nodes

    def __getitem__(self, index: int) -> List[int]:
        """_summary_

        Args:
            index (int): 要素番号

        Returns:
            List[int]: 要素を構成する節点番号
        """
        return self._element_nodes[index]
