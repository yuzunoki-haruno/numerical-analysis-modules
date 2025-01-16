from typing import List


class BoundaryCondition:
    """境界条件管理用クラス

    Raises:
        ValueError: 不正な入力した場合に発生
    """

    DIRICHLET = "dirichlet"
    """Dirichlet境界条件のラベル用文字列"""

    NEUMANN = "neumann"
    """Dirichlet境界条件として扱う文字列のセット"""

    DIRICHLET_STRINGS = {"d", "dirichlet"}
    """Neumann境界条件のラベル用文字列"""

    NEUMANN_STRINGS = {"n", "neumann"}
    """Neumann境界条件として扱う文字列のセット"""

    @classmethod
    def from_string(cls, string: str) -> str:
        """文字列を境界条件管理用ラベルに変換する関数

        Args:
            string (str): 変換前の文字列

        Raises:
            ValueError: ラベルに変換できない文字列を入力した場合に発生

        Returns:
            str: ラベル
        """
        string = string.lower()
        if string in cls.DIRICHLET_STRINGS:
            return cls.DIRICHLET
        elif string in cls.NEUMANN_STRINGS:
            return cls.NEUMANN
        else:
            message = "An invalid string was entered. The strings that can be entered is as follows:\n"
            message += f"  - BoundaryCondition.DIRICHLET: {BoundaryCondition.DIRICHLET_STRINGS}\n"
            message += f"  - BoundaryCondition.NEUMANN  : {BoundaryCondition.NEUMANN_STRINGS}"
            raise ValueError(message)

    @classmethod
    def from_strings(cls, strings: List[str]) -> List[str]:
        """文字列のリストを境界条件ラベルのリストに変換する関数

        Args:
            strings (List[str]): 文字列リスト

        Returns:
            List[str]: 境界条件ラベルのリスト
        """
        return [cls.from_string(s) for s in strings]

    @classmethod
    def to_indices(cls, condition: str, strings: List[str]) -> List[int]:
        """指定した境界条件が課された要素のインデックスを取得する関数

        Args:
            condition (str): 検索対象の境界条件ラベル
            strings (List[str]): 境界条件ラベルのリスト

        Returns:
            List[int]: 境界条件`condition`が課されたインデックスのリスト
        """
        condition = cls.from_string(condition)
        return [i for i, s in enumerate(strings) if cls.from_string(s) == condition]

    @classmethod
    def to_mask(cls, condition: str, strings: List[str]) -> List[bool]:
        """指定した境界条件が課された要素のマスクを取得する関数

        Args:
            condition (str): 検索対象の境界条件ラベル
            strings (List[str]): 境界条件ラベルのリスト

        Returns:
            List[bool]: 境界条件`condition`のマスク
        """
        condition = cls.from_string(condition)
        return [cls.from_string(s) == condition for s in strings]

    @classmethod
    def check_size(cls, conditions: List[str], size: int) -> bool:
        if len(conditions) == size:
            return True
        else:
            message = f"The length of the list `conditions` must be {size}, but it is {len(conditions)}."
            raise IndexError(message)
