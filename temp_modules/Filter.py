import numpy as np
from scipy.linalg import expm

from LieGroup import SE3


class ExtendedKalmanFilter:
    def __init__(self, initial_state, initial_covariance, process_noise, measurement_noise):
        """
        初始化 EKF。

        参数:
            initial_state (np.ndarray): 初始状态向量 (n x 1).
            initial_covariance (np.ndarray): 初始状态协方差矩阵 (n x n).
            process_noise (np.ndarray): 过程噪声协方差矩阵 (n x n).
            measurement_noise (np.ndarray): 观测噪声协方差矩阵 (m x m).
        """
        self.state = initial_state  # 状态向量
        self.covariance = initial_covariance  # 状态协方差矩阵
        self.process_noise = process_noise  # 过程噪声协方差矩阵
        self.measurement_noise = measurement_noise  # 观测噪声协方差矩阵

    def predict(self, control_input=None):
        """
        预测步骤。

        参数:
            control_input (np.ndarray): 控制输入向量 (可选).
        """
        # 使用系统模型预测状态
        self.state = self.system_model(self.state, control_input)

        # 计算系统模型的雅可比矩阵
        F = self.system_jacobian(self.state, control_input)

        # 更新状态协方差矩阵
        self.covariance = F @ self.covariance @ F.T + self.process_noise

    def update(self, measurement):
        """
        更新步骤。

        参数:
            measurement (np.ndarray): 观测值向量 (m x 1).
        """
        # 使用观测模型预测观测值
        predicted_measurement = self.measurement_model(self.state)

        # 计算观测模型的雅可比矩阵
        H = self.measurement_jacobian(self.state)

        # 计算观测残差
        measurement_residual = measurement - predicted_measurement

        # 计算残差协方差矩阵
        S = H @ self.covariance @ H.T + self.measurement_noise

        # 计算卡尔曼增益
        K = self.covariance @ H.T @ np.linalg.inv(S)

        # 更新状态和协方差矩阵
        self.state = self.state + K @ measurement_residual
        self.covariance = (np.eye(len(self.state)) - K @ H) @ self.covariance

    def system_model(self, state, control_input=None):
        """
        系统模型（状态转移函数）。需要用户自定义。

        参数:
            state (np.ndarray): 当前状态向量.
            control_input (np.ndarray): 控制输入向量 (可选).

        返回:
            np.ndarray: 预测的状态向量.
        """
        raise NotImplementedError("系统模型需要用户自定义。")

    def system_jacobian(self, state, control_input=None):
        """
        系统模型的雅可比矩阵。需要用户自定义。

        参数:
            state (np.ndarray): 当前状态向量.
            control_input (np.ndarray): 控制输入向量 (可选).

        返回:
            np.ndarray: 系统模型的雅可比矩阵.
        """
        raise NotImplementedError("系统模型的雅可比矩阵需要用户自定义。")

    def measurement_model(self, state):
        """
        观测模型（观测函数）。需要用户自定义。

        参数:
            state (np.ndarray): 当前状态向量.

        返回:
            np.ndarray: 预测的观测值向量.
        """
        raise NotImplementedError("观测模型需要用户自定义。")

    def measurement_jacobian(self, state):
        """
        观测模型的雅可比矩阵。需要用户自定义。

        参数:
            state (np.ndarray): 当前状态向量.

        返回:
            np.ndarray: 观测模型的雅可比矩阵.
        """
        raise NotImplementedError("观测模型的雅可比矩阵需要用户自定义。")


class SimpleEKF(ExtendedKalmanFilter):
    def __init__(self, initial_state, initial_covariance, process_noise, measurement_noise, dt):
        super().__init__(initial_state, initial_covariance, process_noise, measurement_noise)
        self.dt = dt  # 时间步长

    def system_model(self, state: np.ndarray, control_input: np.ndarray = None):
        """
        系统模型：状态随时间的演化。

        参数:
            state (np.ndarray): 当前状态向量 [x, y, w, h, vx, vy, vw, vh]。
            control_input (np.ndarray): 控制输入向量（可选）。

        返回:
            np.ndarray: 预测的状态向量。
        """
        # 分解状态
        p = state[:4]  # 位置 [x, y, w, h]
        v = state[4:]  # 速度 [vx, vy, vw, vh]

        # 更新位置
        p_new = p + v * self.dt

        # 返回新状态
        return np.hstack([p_new, v])

    def system_jacobian(self, state: np.ndarray, control_input: np.ndarray = None):
        """
        系统模型的雅可比矩阵。

        参数:
            state (np.ndarray): 当前状态向量 [x, y, w, h, vx, vy, vw, vh]。
            control_input (np.ndarray): 控制输入向量（可选）。

        返回:
            np.ndarray: 系统模型的雅可比矩阵 (8x8)。
        """
        return np.eye(8)  # 对于简单的线性模型，雅可比矩阵是单位矩阵

    def measurement_model(self, state: np.ndarray):
        """
        观测模型：从状态到观测值的映射。

        参数:
            state (np.ndarray): 当前状态向量 [x, y, w, h, vx, vy, vw, vh]。

        返回:
            np.ndarray: 预测的观测值向量 [x, y, w, h]。
        """
        # 提取位置信息
        p = state[:4]  # [x, y, w, h]
        return p

    def measurement_jacobian(self, state: np.ndarray):
        """
        观测模型的雅可比矩阵。

        参数:
            state (np.ndarray): 当前状态向量 [x, y, w, h, vx, vy, vw, vh]。

        返回:
            np.ndarray: 观测模型的雅可比矩阵 (4x8)。
        """
        # 观测模型对状态的雅可比矩阵
        H = np.zeros((4, 8))
        H[:4, :4] = np.eye(4)  # 观测值只与位置相关
        return H


class SE3EKF:
    def __init__(self, initial_state: np.ndarray, initial_covariance: np.ndarray, process_noise: np.ndarray,
                 measurement_noise: np.ndarray, dt: float):
        """
        初始化 SE(3) 的 EKF。

        参数:
            initial_state (np.ndarray): 初始状态向量 (6x1, 李代数形式).
            initial_covariance (np.ndarray): 初始状态协方差矩阵 (6x6).
            process_noise (np.ndarray): 过程噪声协方差矩阵 (6x6).
            measurement_noise (np.ndarray): 观测噪声协方差矩阵 (m x m).
            dt (float): 时间步长.
        """
        self.state = initial_state  # 状态向量 (李代数形式)
        self.covariance = initial_covariance  # 状态协方差矩阵
        self.process_noise = process_noise  # 过程噪声协方差矩阵
        self.measurement_noise = measurement_noise  # 观测噪声协方差矩阵
        self.dt = dt  # 时间步长

    def predict(self, control_input: np.ndarray):
        """
        预测步骤。

        参数:
            control_input (np.ndarray): 控制输入向量 (6x1, [omega; v]).
        """
        # 使用系统模型预测状态
        self.state = self.system_model(self.state, control_input)

        # 计算系统模型的雅可比矩阵
        F = self.system_jacobian(self.state, control_input)

        # 更新状态协方差矩阵
        self.covariance = F @ self.covariance @ F.T + self.process_noise

    def update(self, measurement: np.ndarray):
        """
        更新步骤。

        参数:
            measurement (np.ndarray): 观测值向量 (m x 1).
        """
        # 使用观测模型预测观测值
        predicted_measurement = self.measurement_model(self.state)

        # 计算观测模型的雅可比矩阵
        H = self.measurement_jacobian(self.state)

        # 计算观测残差
        measurement_residual = measurement - predicted_measurement

        # 计算残差协方差矩阵
        S = H @ self.covariance @ H.T + self.measurement_noise

        # 计算卡尔曼增益
        K = self.covariance @ H.T @ np.linalg.inv(S)

        # 更新状态和协方差矩阵
        self.state = self.state + K @ measurement_residual
        self.covariance = (np.eye(6) - K @ H) @ self.covariance

    def system_model(self, state: np.ndarray, control_input: np.ndarray):
        """
        系统模型：状态随时间的演化。

        参数:
            state (np.ndarray): 当前状态向量 (6x1).
            control_input (np.ndarray): 控制输入向量 (6x1, [omega; v]).

        返回:
            np.ndarray: 预测的状态向量.
        """
        return state + control_input * self.dt

    def system_jacobian(self, state: np.ndarray, control_input: np.ndarray):
        """
        系统模型的雅可比矩阵。

        参数:
            state (np.ndarray): 当前状态向量.
            control_input (np.ndarray): 控制输入向量.

        返回:
            np.ndarray: 系统模型的雅可比矩阵.
        """
        return np.eye(6)  # 对于简单的线性模型，雅可比矩阵是单位矩阵

    def measurement_model(self, state: np.ndarray):
        """
        观测模型：从状态到观测值的映射。

        参数:
            state (np.ndarray): 当前状态向量.

        返回:
            np.ndarray: 预测的观测值向量.
        """
        # 假设观测值是位姿变换后的点坐标
        se3 = SE3.from_se3(state)
        point = np.array([1, 0, 0])  # 假设观测一个固定点
        return se3.R @ point + se3.t

    def measurement_jacobian(self, state: np.ndarray):
        """
        观测模型的雅可比矩阵。

        参数:
            state (np.ndarray): 当前状态向量.

        返回:
            np.ndarray: 观测模型的雅可比矩阵.
        """
        # 计算观测模型对状态的雅可比矩阵
        omega = state[:3]
        R = expm(SE3.skew_symmetric(omega))
        J = np.zeros((3, 6))
        J[:3, :3] = R  # 对旋转部分的雅可比
        J[:3, 3:] = np.eye(3)  # 对平移部分的雅可比
        return J


class DetectionToFilterAdapter:
    def __init__(self, initial_velocity=0, initial_covariance=1.0, process_noise=0.1, measurement_noise=1.0, dt=0.1):
        """
        初始化适配器。

        参数:
            initial_velocity: 初始速度（假设为 0）。
            initial_covariance: 初始状态协方差。
            process_noise: 过程噪声。
            measurement_noise: 观测噪声。
            dt: 时间步长。
        """
        self.initial_velocity = initial_velocity
        self.initial_covariance = initial_covariance
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.dt = dt

    def convert(self, detection):
        """
        将检测结果转换为滤波器的初始化状态。

        参数:
            detection: 检测结果 [x, y, w, h]。

        返回:
            dict: 包含滤波器初始化状态的字典。
        """
        x, y, w, h = detection

        # 初始化状态向量 [x, y, w, h, vx, vy, vw, vh]
        state = np.array(
            [x, y, w, h, self.initial_velocity, self.initial_velocity, self.initial_velocity, self.initial_velocity])

        # 初始化状态协方差矩阵
        covariance = np.eye(8) * self.initial_covariance

        # 过程噪声和观测噪声
        process_noise = np.eye(8) * self.process_noise
        measurement_noise = np.eye(4) * self.measurement_noise

        return {
            'state': state,
            'initial_covariance': covariance,
            'process_noise': process_noise,
            'measurement_noise': measurement_noise,
            'dt': self.dt
        }


def user_case_simple_kf():
    # 初始化状态和参数
    initial_state = np.array([0, 1])  # [位置, 速度]
    initial_covariance = np.eye(2) * 0.1
    process_noise = np.eye(2) * 0.01
    measurement_noise = np.eye(1) * 0.1
    dt = 0.1  # 时间步长

    # 创建 EKF 实例
    ekf = SimpleEKF(initial_state, initial_covariance, process_noise, measurement_noise, dt)

    # 模拟数据
    true_state = np.array([0, 1])  # 真实状态
    measurements = [true_state[0] ** 2 + np.random.normal(0, 0.1) for _ in range(10)]  # 观测值

    # 运行 EKF
    for z in measurements:
        ekf.predict()
        ekf.update(np.array([z]))
        print("更新后的状态:", ekf.state)


def user_case_ekf():
    # 初始化状态和参数
    initial_state = np.zeros(6)  # 初始状态 (李代数形式)
    initial_covariance = np.eye(6) * 0.1  # 初始协方差矩阵
    process_noise = np.eye(6) * 0.01  # 过程噪声协方差矩阵
    measurement_noise = np.eye(3) * 0.1  # 观测噪声协方差矩阵
    dt = 0.1  # 时间步长

    # 创建 SE(3) EKF 实例
    ekf = SE3EKF(initial_state, initial_covariance, process_noise, measurement_noise, dt)

    # 模拟数据
    control_input = np.array([0.1, 0.2, 0.3, 1, 2, 3])  # 控制输入 ([omega; v])
    measurements = [np.array([1, 0, 0]) + np.random.normal(0, 0.1, 3) for _ in range(10)]  # 观测值

    # 运行 EKF
    for z in measurements:
        ekf.predict(control_input)
        ekf.update(z)
        print("更新后的状态:", ekf.state)


def user_case_det_adpt():
    # 模拟检测结果
    detection = [100, 100, 50, 50]  # [x, y, w, h]

    # 创建适配器
    adapter = DetectionToFilterAdapter()

    # 将检测结果转换为滤波器初始化状态
    filter_init = adapter.convert(detection)

    # 打印滤波器初始化状态
    print("滤波器初始化状态:", filter_init)


# 示例使用
if __name__ == "__main__":
    user_case_det_adpt()
