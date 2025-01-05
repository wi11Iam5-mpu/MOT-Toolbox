import numpy as np
from scipy.linalg import expm, logm


class SE3:
    def __init__(self, rotation: np.ndarray, translation: np.ndarray):
        """
        初始化 SE(3) 变换矩阵。

        参数:
            rotation (np.ndarray): 3x3 旋转矩阵 (R).
            translation (np.ndarray): 3x1 平移向量 (t).
        """
        assert rotation.shape == (3, 3), "Rotation matrix must be 3x3."
        assert translation.shape == (3,), "Translation vector must be 3x1."

        self.R = rotation
        self.t = translation
        self.T = self._construct_matrix()  # 4x4 齐次变换矩阵T, 表示三维刚体位姿 (pose)

    def _construct_matrix(self):
        """
        构造 4x4 齐次变换矩阵。
        """
        matrix = np.eye(4)
        matrix[:3, :3] = self.R
        matrix[:3, 3] = self.t
        return matrix

    def transform_point(self, point: np.ndarray):
        """
        对点进行 SE(3) 变换。

        参数:
            point (np.ndarray): 3x1 的点坐标.

        返回:
            np.ndarray: 变换后的点坐标.
        """
        assert point.shape == (3,), "Point must be a 3x1 vector."
        return self.R @ point + self.t

    def inverse(self):
        """
        计算 SE(3) 的逆变换。

        返回:
            SE3: 逆变换的 SE(3) 对象.
        """
        inv_rotation = self.R.T
        inv_translation = -inv_rotation @ self.t
        return SE3(inv_rotation, inv_translation)

    def compose(self, other):
        """
        组合两个 SE(3) 变换。

        参数:
            other (SE3): 另一个 SE(3) 变换.

        返回:
            SE3: 组合后的 SE(3) 变换.
        """
        new_rotation = self.R @ other.R
        new_translation = self.R @ other.t + self.t
        return SE3(new_rotation, new_translation)

    @classmethod
    def from_se3(self, twist: np.ndarray):
        """
        从李代数 se(3) 构造 SE(3) 变换（指数映射）。

        参数:
            twist (np.ndarray): 6x1 的李代数向量 [phi; v].

        返回:
            SE3: 对应的 SE(3) 变换.
        """
        assert twist.shape == (6,), "Twist must be a 6x1 vector."

        phi = twist[:3]
        v = twist[3:]

        # 构造 se(3) 的矩阵形式
        xi = np.zeros((4, 4))
        xi[:3, :3] = self.skew_symmetric(phi)
        xi[:3, 3] = v

        # 指数映射
        T = expm(xi)
        return SE3(T[:3, :3], T[:3, 3])

    def to_se3(self):
        """
        将 SE(3) 变换映射到李代数 se(3)（对数映射）。

        返回:
            np.ndarray: 6x1 的李代数向量 [phi; v].
        """
        # 对数映射
        xi = logm(self.T)

        # 提取 phi 和 v
        phi = self.unskew_symmetric(xi[:3, :3])
        v = xi[:3, 3]
        return np.hstack([phi, v])

    @staticmethod
    def skew_symmetric(phi: np.ndarray):
        """
        将角速度 phi 转换为反对称矩阵。
        """
        return np.array([
            [0, -phi[2], phi[1]],
            [phi[2], 0, -phi[0]],
            [-phi[1], phi[0], 0]
        ])

    @staticmethod
    def unskew_symmetric(matrix: np.ndarray):
        """
        将反对称矩阵转换回角速度 phi。
        """
        return np.array([matrix[2, 1], matrix[0, 2], matrix[1, 0]])

    def __repr__(self):
        return f"SE3(\nRotation:\n{self.R}\nTranslation:\n{self.t}\n)"


# 示例用法
if __name__ == "__main__":
    # 定义一个旋转矩阵和平移向量 R, t
    R = np.array([[0, -1, 0],
                  [1, 0, 0],
                  [0, 0, 1]])
    t = np.array([1, 2, 3])

    # 创建 SE(3) 对象
    obj1 = SE3(R, t)
    print("SE(3) 变换矩阵:\n", obj1.T)

    # 变换一个点
    point = np.array([1, 0, 0])
    transformed_point = obj1.transform_point(point)
    print("变换后的点:", transformed_point)

    # 计算逆变换
    inv_se3 = obj1.inverse()
    print("逆变换矩阵:\n", inv_se3.T)

    # 组合两个 SE(3) 变换
    se3_2 = SE3(np.eye(3), np.array([4, 5, 6]))
    composed_se3 = obj1.compose(se3_2)
    print("组合后的变换矩阵:\n", composed_se3.T)

    # 定义一个李代数 twist
    twist = np.array([0.1, 0.2, 0.3, 1, 2, 3])  # [phi; v]

    # 从李代数构造 SE(3)
    obj2 = SE3.from_se3(twist)
    print("从李代数构造的 SE(3) 变换矩阵:\n", obj2.T)

    # 从 SE(3) 映射回李代数
    recovered_twist = obj2.to_se3()
    print("恢复的李代数 twist:\n", recovered_twist)
