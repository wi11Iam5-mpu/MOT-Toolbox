import numpy as np

from Filter import SimpleEKF, DetectionToFilterAdapter
from DetectionReader import MOT17DetectionReader, MOT17TrackSaver
from Tracker import MultiObjectTracker, evaluate_mot17
from utils import Visualizer

is_vis = True  # True False

# MOT17 det.txt 文件路径
file_path = r"O:\G\files\MHT\GMOT\trackers\data\mot17_train_byte\MOT17-02-FRCNN\det\det.txt"
# 输出跟踪结果文件路径
output_path = output_file = r"O:\G\Projects\TrackEval-master\TrackEval-master\data\trackers\mot_challenge\MOT17-train\MHT_tracklet\data\MOT17-02-FRCNN.txt"

# 创建 MOT17 检测结果读取器
reader = MOT17DetectionReader(file_path, confidence_threshold=0.5)
saver = MOT17TrackSaver(output_path)
# 创建适配器
adapter = DetectionToFilterAdapter()
# 创建多目标跟踪器
tracker = MultiObjectTracker(filter_class=SimpleEKF, adapter=adapter)

# 创建可视化工具
visualizer = Visualizer(vis_mode=["center"])
# 创建空白图像
image = np.zeros((1080, 1920, 3), dtype=np.uint8)

# 逐帧处理检测结果
for frame in range(0, len(reader)):
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
        visualizer.save_frame(image, "output_frame.png")
        visualizer.save_video([image], "output_video.mp4", fps=30)

    # 打印当前轨迹
    print(f"Frame {frame}:")
    for track in tracker.tracks:
        print(f"  Track ID: {track.track_id}, State: {track.get_state()}")
# 关闭文件
saver.close()
evaluate_mot17()