import numpy as np


class MOT17DetectionReader:
    def __init__(self, file_path):
        """
        初始化 MOT17 检测结果读取器。

        参数:
            file_path (str): MOT17 det.txt 文件的路径。
        """
        self.file_path = file_path
        self.detections = self._load_detections()

    def _load_detections(self):
        """
        加载 MOT17 检测结果。

        返回:
            dict: 按帧号索引的检测结果列表。
        """
        detections = {}
        with open(self.file_path, 'r') as file:
            for line in file:
                # 解析每一行
                parts = line.strip().split(',')
                frame = int(parts[0])  # 帧号
                x, y, w, h = map(float, parts[2:6])  # 边界框
                conf = float(parts[6])  # 置信度

                # 将检测结果添加到对应帧的列表中
                if frame not in detections:
                    detections[frame] = []
                detections[frame].append([x, y, w, h, conf])

        return detections

    def get_detections_by_frame(self, frame):
        """
        获取指定帧的检测结果。

        参数:
            frame (int): 帧号。

        返回:
            list: 检测结果列表，格式为 [x, y, w, h]。
        """
        if frame in self.detections:
            return [det[:4] for det in self.detections[frame]]  # 只返回 [x, y, w, h]
        else:
            return []

    def __len__(self):
        return len(self.detections)


class MOT17TrackSaver:
    def __init__(self, output_path):
        """
        初始化轨迹保存适配器。

        参数:
            output_path (str): 输出文件路径。
        """
        self.output_path = output_path
        self.file = open(output_path, 'w')  # 打开文件

    def save_tracks(self, frame_id, tracks):
        """
        保存当前帧的轨迹结果。

        参数:
            frame_id (int): 当前帧号。
            tracks (list): 当前帧的轨迹列表。
        """
        for track in tracks:
            track_id = track.track_id
            state = track.get_state()  # 获取当前状态 [x, y, w, h]
            x, y, w, h = map(float, state[:4])  # 转换为浮点数

            # 写入 MOT17 格式的轨迹结果
            line = f"{frame_id},{track_id},{x:.2f},{y:.2f},{w:.2f},{h:.2f},1,-1,-1,-1\n"
            self.file.write(line)

    def close(self):
        """
        关闭文件，确保数据写入磁盘。
        """
        self.file.close()


def user_case_det_reader():
    # MOT17 det.txt 文件路径
    file_path = r"O:\G\files\MHT\GMOT\trackers\data\mot17_train_byte\MOT17-02-FRCNN\det\det.txt"
    # 创建 MOT17 检测结果读取器
    reader = MOT17DetectionReader(file_path)
    print("总帧数:", len(reader))
    # 获取第 1 帧的检测结果
    frame_1_detections = reader.get_detections_by_frame(1)
    print("第 1 帧的检测结果:", frame_1_detections)
    # 获取第 2 帧的检测结果
    frame_2_detections = reader.get_detections_by_frame(2)
    print("第 2 帧的检测结果:", frame_2_detections)


def user_case_track_saver():
    global tracks

    # 模拟轨迹数据
    class Track:
        def __init__(self, track_id, state):
            self.track_id = track_id
            self.state = state

        def get_state(self):
            return self.state

    tracks = [
        Track(1, [100.5, 100.5, 50.5, 50.5]),  # [x, y, w, h]
        Track(2, [200.5, 200.5, 60.5, 60.5]),
        Track(3, [300.5, 300.5, 70.5, 70.5])
    ]
    # 创建轨迹保存适配器
    output_path = "output_tracks.txt"
    saver = MOT17TrackSaver(output_path)
    # 保存第 1 帧的轨迹
    saver.save_tracks(1, tracks)
    # 保存第 2 帧的轨迹
    saver.save_tracks(2, tracks)
    # 关闭文件
    saver.close()
    print(f"轨迹结果已保存到 {output_path}")


# 示例使用
if __name__ == "__main__":
    user_case_det_reader()
    user_case_track_saver()
