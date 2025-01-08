import numpy as np
from collections import defaultdict
from scipy.optimize import linear_sum_assignment


class MultiObjectTracker:
    def __init__(self, filter_class, adapter, max_age=30, min_hits=3, iou_threshold=0.3):
        """
        初始化多目标跟踪器。

        参数:
            filter_class: 滤波器类（如卡尔曼滤波器）。
            adapter: 检测结果到滤波器的适配器。
            max_age (int): 轨迹的最大丢失帧数。
            min_hits (int): 轨迹被确认所需的最小连续匹配帧数。
            iou_threshold (float): 数据关联的 IOU 阈值。
        """
        self.filter_class = filter_class
        self.adapter = adapter
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold

        self.tracks = []  # 当前活跃的轨迹
        self.frame_count = 0  # 帧计数器
        self.next_id = 1  # 下一个轨迹的 ID

    def update(self, detections):
        """
        更新跟踪器状态。

        参数:
            detections (list): 当前帧的检测结果，每个检测结果为 [x, y, w, h]。
        """
        self.frame_count += 1

        # 预测所有轨迹的状态
        for track in self.tracks:
            track.predict()

        # 数据关联：将检测结果与轨迹匹配
        matched_indices, unmatched_detections, unmatched_tracks = self._data_association(detections)

        # 更新匹配的轨迹
        for track_idx, detection_idx in matched_indices:
            self.tracks[track_idx].update(detections[detection_idx])

        # 处理未匹配的检测结果（创建新轨迹）
        for detection_idx in unmatched_detections:
            self._create_track(detections[detection_idx])

        # 处理未匹配的轨迹（标记为丢失）
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()

        # 删除丢失的轨迹
        self.tracks = [track for track in self.tracks if not track.is_deleted(self.max_age)]

    def _data_association(self, detections):
        """
        数据关联：将检测结果与轨迹匹配。

        参数:
            detections (list): 当前帧的检测结果。

        返回:
            matched_indices: 匹配的 (轨迹索引, 检测索引) 列表。
            unmatched_detections: 未匹配的检测索引列表。
            unmatched_tracks: 未匹配的轨迹索引列表。
        """
        if len(self.tracks) == 0:
            return [], list(range(len(detections))), []

        # 计算 IOU 矩阵
        iou_matrix = np.zeros((len(self.tracks), len(detections)), dtype=np.float32)
        for t, track in enumerate(self.tracks):
            for d, detection in enumerate(detections):
                iou_matrix[t, d] = self._iou(track.get_state(), detection)

        # 使用匈牙利算法进行匹配
        matched_indices = linear_sum_assignment(-iou_matrix)
        matched_indices = list(zip(matched_indices[0], matched_indices[1]))

        # 过滤低 IOU 的匹配
        matched_indices = [(t, d) for t, d in matched_indices if iou_matrix[t, d] >= self.iou_threshold]

        # 找出未匹配的检测和轨迹
        unmatched_detections = set(range(len(detections))) - set(d for _, d in matched_indices)
        unmatched_tracks = set(range(len(self.tracks))) - set(t for t, _ in matched_indices)

        return matched_indices, list(unmatched_detections), list(unmatched_tracks)

    def _create_track(self, detection):
        """
        创建新轨迹。

        参数:
            detection: 检测结果。
        """
        # 使用适配器将检测结果转换为滤波器初始化状态
        filter_init = self.adapter.convert(detection)

        # 创建新轨迹
        track = Track(self.next_id, self.filter_class, filter_init)
        self.tracks.append(track)
        self.next_id += 1

    @staticmethod
    def _iou(box1, box2):
        """
        计算两个边界框的 IOU。

        参数:
            box1: 第一个边界框 [x1, y1, w1, h1]。
            box2: 第二个边界框 [x2, y2, w2, h2]。

        返回:
            float: IOU 值。
        """
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        # 计算交集区域
        x_inter = max(x1, x2)
        y_inter = max(y1, y2)
        w_inter = max(0, min(x1 + w1, x2 + w2) - x_inter)
        h_inter = max(0, min(y1 + h1, y2 + h2) - y_inter)
        area_inter = w_inter * h_inter

        # 计算并集区域
        area1 = w1 * h1
        area2 = w2 * h2
        area_union = area1 + area2 - area_inter

        return area_inter / area_union if area_union > 0 else 0


class Track:
    def __init__(self, track_id, filter_class, filter_init):
        """
        初始化轨迹。

        参数:
            detection: 检测结果。
            track_id (int): 轨迹 ID。
            filter_class: 滤波器类。
        """
        self.track_id = track_id

        # 从 filter_init 中提取参数
        state = filter_init['state']
        initial_covariance = filter_init['initial_covariance']
        process_noise = filter_init['process_noise']
        measurement_noise = filter_init['measurement_noise']
        dt = filter_init['dt']
        # 初始化滤波器
        self.filter = filter_class(
            state, initial_covariance, process_noise, measurement_noise, dt
        )
        self.hits = 1  # 连续匹配帧数
        self.age = 1  # 轨迹年龄
        self.time_since_update = 0  # 自上次更新以来的帧数

    def predict(self):
        """预测轨迹状态。"""
        self.filter.predict()
        self.age += 1
        self.time_since_update += 1

    def update(self, detection):
        """更新轨迹状态。"""
        self.filter.update(detection)
        self.hits += 1
        self.time_since_update = 0

    def mark_missed(self):
        """标记轨迹为丢失。"""
        self.time_since_update += 1

    def is_deleted(self, max_age=30):
        """判断轨迹是否应被删除。"""
        return self.time_since_update > max_age  # 超过 3 帧未匹配则删除

    def get_state(self):
        """获取轨迹的当前状态（边界框）。"""
        return self.filter.state[:4]  # 假设状态为 [x, y, w, h]


# ===== 临时的评估代码 =====
def evaluate_mot17():
    import subprocess
    # 设置命令和工作目录
    command = ["python", "run_mot_challenge.py"]
    working_directory = r"O:\G\Projects\TrackEval-master\TrackEval-master\scripts"

    # 使用 subprocess 运行命令
    try:
        result = subprocess.run(command, cwd=working_directory, capture_output=True, text=True, check=True)
        print("Command executed successfully!")
        print("Output:")
        print(result.stdout)  # 打印脚本的标准输出
    except subprocess.CalledProcessError as e:
        print("Error occurred while executing the command:")
        print(e.stderr)  # 打印脚本的错误输出


# 示例使用
if __name__ == "__main__":
    from Filter import SimpleKalmanFilter, DetectionToFilterAdapter
    from DetectionReader import MOT17DetectionReader, MOT17TrackSaver
    from utils import Visualizer

    is_vis = True

    # MOT17 det.txt 文件路径
    file_path = r"O:\G\files\MHT\GMOT\trackers\data\mot17_train_byte\MOT17-02-FRCNN\det\det.txt"
    # 输出跟踪结果文件路径
    output_path = r"O:\G\Projects\TrackEval-master\TrackEval-master\data\trackers\mot_challenge\MOT17-train\MHT_tracklet\data\MOT17-02-FRCNN.txt"

    # 创建 MOT17 检测结果读取器
    reader = MOT17DetectionReader(file_path, confidence_threshold=0.7)  # 0.5 - 0.6 - 0.7 提升很大, 忍耐窗口3->30，稍微降低MOTA, 但是提升了IDF1
    saver = MOT17TrackSaver(output_path)

    # TODO:adapter和filter的参数应该绑定
    # 创建适配器
    adapter = DetectionToFilterAdapter(filter_type='simple')
    # 创建多目标跟踪器
    tracker = MultiObjectTracker(filter_class=SimpleKalmanFilter, adapter=adapter)

    # 创建可视化工具
    visualizer = Visualizer(vis_mode=["center", "bottom_center"])
    # 创建空白图像
    image = np.zeros((1080, 1920, 3), dtype=np.uint8)

    # 逐帧处理检测结果
    for frame in range(1, len(reader)):  # 假设处理前 100 帧
        # 获取当前帧的检测结果
        detections = reader.get_detections_by_frame(frame)
        # 更新跟踪器
        tracker.update(detections)
        saver.save_tracks(frame, tracker.tracks)
        if is_vis:
            # 绘制检测结果和轨迹
            # image = visualizer.draw_detections(image, detections)
            image = visualizer.draw_tracks(image, tracker.tracks)
            # 显示结果
            visualizer.show_frame(image)

            # 保存结果
            # visualizer.save_frame(image, "output_frame.png")
            # visualizer.save_video([image], "output_video.mp4", fps=30)

        # 打印当前轨迹
        print(f"Frame {frame}:")
        for track in tracker.tracks:
            print(f"  Track ID: {track.track_id}, State: {track.get_state()}")
    # 关闭文件
    saver.close()
    evaluate_mot17()
