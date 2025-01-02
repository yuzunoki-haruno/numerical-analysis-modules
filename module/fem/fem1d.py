import numpy as np
from numpy.typing import NDArray
from scipy.sparse import lil_matrix

from module.discretization import BoundaryCondition, LineMesh, LineMeshHighOrder


class Fem1d:
    """一次元有限要素法"""

    def __init__(self, mesh: LineMesh | LineMeshHighOrder) -> None:
        """一次元有限要素法

        Args:
            mesh (LineMesh | LineMeshHighOrder): メッシュデータ

        Raises:
            ValueError: 不正なメッシュデータを入力した場合に発生
        """
        self.mesh = mesh
        if isinstance(mesh, LineMesh):
            self._laplacian = _laplacian_matrix(mesh)
            self._term = _term_matrix(mesh)
        elif isinstance(mesh, LineMeshHighOrder):
            self._laplacian = _laplacian_matrix_high_order(mesh)
            self._term = _term_matrix_high_order(mesh)
        else:
            raise ValueError

    @property
    def laplacian_matrix(self) -> lil_matrix:
        """Laplace作用素に対応する行列

        Returns:
            lil_matrix: _description_
        """
        return self._laplacian.copy()

    @property
    def term_matrix(self) -> lil_matrix:
        """一般的な項に対応する行列

        Returns:
            lil_matrix: _description_
        """
        return self._term.copy()

    def laplacian(self, vec: NDArray) -> NDArray:
        """ラプラス作用素を適用する関数

        Args:
            vec (NDArray): 関数値データ

        Returns:
            NDArray: ラプラス作用素を適用した結果の離散データ
        """
        return np.array(self._laplacian.dot(vec))

    def term(self, vec: NDArray) -> NDArray:
        """一般的な項の離散データを計算する関数

        Args:
            vec (NDArray): 関数値データ

        Returns:
            NDArray: 一般的な項の離散データ
        """
        return np.array(self._term.dot(vec))

    def implement_dirichlet(self, coefficient: lil_matrix, rhs: NDArray, values: NDArray) -> None:
        """係数行列および右辺ベクトルにDirichlet境界条件を課す関数

        Args:
            coefficient (lil_matrix): 係数行列
            rhs (NDArray): 右辺ベクトル
            values (NDArray): 境界値データ
        """
        local_index = BoundaryCondition.to_indices(BoundaryCondition.DIRICHLET, self.mesh.conditions)
        global_index = [self.mesh.boundary_nodes[i] for i in local_index]
        d = np.zeros_like(rhs)
        d[global_index] = values[global_index]
        rhs -= coefficient.dot(d)
        rhs[global_index] = values[global_index]
        coefficient[global_index, :] = 0.0
        coefficient[:, global_index] = 0.0
        coefficient[global_index, global_index] = 1.0

    def implement_neumann(self, rhs: NDArray, values: NDArray) -> None:
        """右辺ベクトルにNeumann境界条件を課す関数

        Args:
            rhs (NDArray): 右辺ベクトル
            values (NDArray): 境界値データ
        """
        local_index = BoundaryCondition.to_indices(BoundaryCondition.NEUMANN, self.mesh.conditions)
        global_index = [self.mesh.boundary_nodes[i] for i in local_index]
        for i, m in zip(global_index, local_index):
            rhs[i] += self.mesh.unit_normals[m] * values[i]


def _laplacian_matrix(mesh: LineMesh) -> lil_matrix:
    """Laplace作用素に対応する行列（一次要素）

    Args:
        mesh (LineMesh): メッシュデータ

    Returns:
        lil_matrix: Laplace作用素に対応する行列
    """
    n_node = mesh.n_node
    matrix = lil_matrix((n_node, n_node))
    for i, j in mesh.element_nodes:
        h = mesh.x[j] - mesh.x[i]
        matrix[i, i] += 1 / h
        matrix[j, j] += 1 / h
        matrix[i, j] -= 1 / h
        matrix[j, i] -= 1 / h
    return matrix


def _term_matrix(mesh: LineMesh) -> lil_matrix:
    """一般的な項に対応する行列（一次要素）

    Args:
        mesh (LineMesh): メッシュデータ

    Returns:
        lil_matrix: 一般的な項に対応する行列
    """
    n_node = mesh.n_node
    matrix = lil_matrix((n_node, n_node))
    for i, j in mesh.element_nodes:
        h = mesh.x[j] - mesh.x[i]
        matrix[i, i] += h / 3
        matrix[j, j] += h / 3
        matrix[i, j] += h / 6
        matrix[j, i] += h / 6
    return matrix


def _laplacian_matrix_high_order(mesh: LineMeshHighOrder) -> lil_matrix:
    """Laplace作用素に対応する行列（二次要素）

    Args:
        mesh (LineMesh): メッシュデータ

    Returns:
        lil_matrix: Laplace作用素に対応する行列
    """
    n_node = mesh.n_node
    matrix = lil_matrix((n_node, n_node))
    for i, j, k in mesh.element_nodes:
        h = mesh.x[j] - mesh.x[i]
        matrix[i, i] += 7 / 3 / h
        matrix[i, j] += 1 / 3 / h
        matrix[i, k] -= 8 / 3 / h
        matrix[j, i] += 1 / 3 / h
        matrix[j, j] += 7 / 3 / h
        matrix[j, k] -= 8 / 3 / h
        matrix[k, i] -= 8 / 3 / h
        matrix[k, j] -= 8 / 3 / h
        matrix[k, k] += 16 / 3 / h
    return matrix


def _term_matrix_high_order(mesh: LineMeshHighOrder) -> lil_matrix:
    """一般的な項に対応する行列（二次要素）

    Args:
        mesh (LineMesh): メッシュデータ

    Returns:
        lil_matrix: 一般的な項に対応する行列
    """
    n_node = mesh.n_node
    matrix = lil_matrix((n_node, n_node))
    for i, j, k in mesh.element_nodes:
        h = mesh.x[j] - mesh.x[i]
        matrix[i, i] += h * 2 / 15
        matrix[i, j] -= h / 30
        matrix[i, k] += h / 15
        matrix[j, i] -= h / 30
        matrix[j, j] += h * 2 / 15
        matrix[j, k] += h / 15
        matrix[k, i] += h / 15
        matrix[k, j] += h / 15
        matrix[k, k] += h * 8 / 15
    return matrix
