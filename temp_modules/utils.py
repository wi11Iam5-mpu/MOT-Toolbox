import cv2
import numpy as np


class Visualizer:
    def __init__(self, colors=None, line_width=2, font_scale=0.5, font_thickness=1):
        """
        初始化可视化工具。

        参数:
            colors (dict): 每个轨迹 ID 对应的颜色。
            line_width (int): 绘制框和轨迹的线宽。
            font_scale (float): 字体大小。
            font_thickness (int): 字体粗细。
        """
        self.colors = colors if colors is not None else {}
        self.line_width = line_width
        self.font_scale = font_scale
        self.font_thickness = font_thickness

    def draw_detections(self, image, detections):
        """
        在图像上绘制检测结果。

        参数:
            image (np.ndarray): 输入图像。
            detections (list): 检测结果列表，每个检测结果为 [x, y, w, h]。

        返回:
            np.ndarray: 绘制检测结果后的图像。
        """
        for det in detections:
            x, y, w, h = map(int, det[:4])  # 转换为整数
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), self.line_width)
        return image

    def draw_tracks(self, image, tracks):
        """
        在图像上绘制目标轨迹。

        参数:
            image (np.ndarray): 输入图像。
            tracks (list): 轨迹列表，每个轨迹包含状态和 ID。

        返回:
            np.ndarray: 绘制轨迹后的图像。
        """
        for track in tracks:
            track_id = track.track_id
            state = track.get_state()  # 获取当前状态 [x, y, w, h]
            x, y, w, h = map(int, state[:4])  # 转换为整数

            # 获取或生成颜色
            if track_id not in self.colors:
                self.colors[track_id] = tuple(np.random.randint(0, 255, 3).tolist())  # 随机颜色

            # 绘制边界框
            cv2.rectangle(image, (x, y), (x + w, y + h), self.colors[track_id], self.line_width)

            # 绘制轨迹 ID
            cv2.putText(image, f"ID: {track_id}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        self.font_scale, self.colors[track_id], self.font_thickness)
        return image

    def show_frame(self, image, window_name="Tracking Result"):
        """
        显示当前帧的可视化结果。

        参数:
            image (np.ndarray): 输入图像。
            window_name (str): 窗口名称。
        """
        cv2.imshow(window_name, image)
        cv2.waitKey(1)  # 等待 1 毫秒

    def save_frame(self, image, output_path):
        """
        将当前帧的可视化结果保存为图像。

        参数:
            image (np.ndarray): 输入图像。
            output_path (str): 输出文件路径。
        """
        cv2.imwrite(output_path, image)

    def save_video(self, frames, output_path, fps=30):
        """
        将多帧可视化结果保存为视频。

        参数:
            frames (list): 多帧图像列表。
            output_path (str): 输出视频文件路径。
            fps (int): 视频帧率。
        """
        if not frames:
            return

        # 获取图像尺寸
        height, width, _ = frames[0].shape

        # 创建视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 视频编码器
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # 写入帧
        for frame in frames:
            out.write(frame)

        # 释放视频写入器
        out.release()


# 示例使用
if __name__ == "__main__":
    # 创建可视化工具
    visualizer = Visualizer()

    # 模拟检测结果和轨迹
    detections = [
        [100, 100, 50, 50],  # [x, y, w, h]
        [200, 200, 60, 60],
        [300, 300, 70, 70]
    ]


    class Track:
        def __init__(self, track_id, state):
            self.track_id = track_id
            self.state = state

        def get_state(self):
            return self.state


    tracks = [
        Track(1, [100, 100, 50, 50]),
        Track(2, [200, 200, 60, 60]),
        Track(3, [300, 300, 70, 70])
    ]

    # 创建空白图像
    image = np.zeros((500, 500, 3), dtype=np.uint8)

    # 绘制检测结果和轨迹
    image = visualizer.draw_detections(image, detections)
    image = visualizer.draw_tracks(image, tracks)

    # 显示结果
    visualizer.show_frame(image)

    # 保存结果
    visualizer.save_frame(image, "output_frame.png")
    visualizer.save_video([image], "output_video.mp4", fps=30)