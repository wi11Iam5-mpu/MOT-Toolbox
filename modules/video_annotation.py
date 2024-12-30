import cv2
import random
# import tkinter as tk
from tkinter import ttk, filedialog
import customtkinter as ctk

from .base_module import BaseModule


class VideoAnnotationModule(BaseModule):
    """视频标注模块：在视频帧中显示对象检测的边界框和标签"""

    def __init__(self, root, name='VideoAnnotationModule', priority=1, show_annotations=False,
                 default_annotation_path="", confidence_threshold=0.5):
        super().__init__(root, name, priority)
        self.show_annotations = ctk.BooleanVar(value=show_annotations)
        self.default_annotation_path = default_annotation_path
        self.confidence_threshold = confidence_threshold
        self.annotations = {}  # 格式：{frame_idx: [(x, y, w, h, confidence, id)]}
        self.tracking_enabled = False  # 是否启用轨迹显示
        self.id_history = {}  # 存储历史 ID 以绘制轨迹

        self.stats = {}  # 用于存储每帧的统计信息
        self.id_colors = {}  # 用于存储每个 id 的颜色

        # 曲线数据
        self.total_boxes_history = []  # 记录总 box 数量
        self.filtered_boxes_history = []  # 记录过滤后的 box 数量

        # 统计信息的覆盖层大小
        self.overlay_width = 300
        self.overlay_height = 150
        self.overlay_font_size = 0.6
        self.resizing = False

        # 初始化字体位置和行间距
        self.label_offset_x = int(0.05 * self.overlay_width)  # 距离面板左侧的偏移量
        self.label_offset_y = int(0.0 * self.overlay_height)  # 距离面板顶部的偏移量
        self.label_spacing = int(0.2 * self.overlay_height)  # 每行文字的间距

        # Bind mouse events for resizing
        self.root.bind("<MouseWheel>", self.resize_with_wheel)

    def _get_color_for_id(self, obj_id):
        """为每个 ID 分配唯一颜色"""
        if obj_id not in self.id_colors:
            # 为新 ID 生成随机颜色
            self.id_colors[obj_id] = (
                random.randint(0, 255),  # R
                random.randint(0, 255),  # G
                random.randint(0, 255),  # B
            )
        return self.id_colors[obj_id]

    def create_controls(self, parent_frame):
        """在控制面板上创建按钮和滑块控件"""
        load_button = ctk.CTkButton(parent_frame, text="Load Annotations", command=self.load_annotations)
        load_button.pack(side=ctk.LEFT, padx=5)

        toggle_checkbox = ctk.CTkCheckBox(parent_frame, text="Show", variable=self.show_annotations)
        toggle_checkbox.pack(side=ctk.LEFT, padx=5)

        # 滑块用于调整置信度阈值
        label = ctk.CTkLabel(parent_frame, text="Confidence Threshold")
        label.pack(side=ctk.LEFT, padx=5)

        self.slider = ctk.CTkSlider(parent_frame, from_=0.0, to=1.0, number_of_steps=100)
        self.slider.set(self.confidence_threshold)
        self.slider.pack(side=ctk.LEFT, padx=5)
        self.slider.bind("<Motion>", self.update_confidence_threshold)

    def update_confidence_threshold(self, event=None, video_player=None):
        """更新置信度阈值并立即刷新画面"""
        self.confidence_threshold = self.slider.get()
        if video_player:  # 如果传入了 VideoPlayer 实例，手动刷新当前帧
            current_frame = video_player.get_current_frame()
            if current_frame is not None:
                processed_frame = self.process_frame(current_frame, video_player.current_frame_idx)
                video_player._display_frame(processed_frame)
                self._precompute_statistics()  # 预计算统计数据

    def load_annotations(self):
        """加载标注文件"""
        annotation_path = filedialog.askopenfilename(initialdir=self.default_annotation_path,
                                                     filetypes=[("Text Files", "*.txt")])
        if annotation_path:
            self.annotations = self._parse_annotations(annotation_path)
            self._check_tracking()  # 检查是否启用轨迹显示
            self._precompute_statistics()  # 预计算统计数据

    def _precompute_statistics(self):
        """预计算每帧的统计数据，减少实时计算开销"""
        self.total_boxes_per_frame = []
        self.filtered_boxes_per_frame = []
        max_frames = max(self.annotations.keys()) if self.annotations else 0

        for frame_idx in range(max_frames + 1):
            if frame_idx in self.annotations:
                total_boxes = len(self.annotations[frame_idx])
                filtered_boxes = sum(
                    1 for _, _, _, _, confidence, _ in self.annotations[frame_idx]
                    if confidence >= self.confidence_threshold
                )
            else:
                total_boxes = 0
                filtered_boxes = 0

            self.total_boxes_per_frame.append(total_boxes)
            self.filtered_boxes_per_frame.append(filtered_boxes)

    def _check_tracking(self):
        """检查是否启用轨迹显示"""
        for frame_idx, boxes in self.annotations.items():
            ids = [box[5] for box in boxes]  # 获取当前帧的所有 ID
            if 1 == len(set(ids)) or len(set(ids)) != len(ids):  # 检查 ID 是否有重复
                self.tracking_enabled = False
                break
        else:
            self.tracking_enabled = True

    def _parse_annotations(self, file_path):
        """解析标注文件"""
        annotations = {}
        with open(file_path, "r") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 7:
                    continue
                frame_idx, obj_id, x, y, w, h, confidence = map(float, parts[:7])
                frame_idx = int(frame_idx)
                obj_id = int(obj_id)
                if frame_idx not in annotations:
                    annotations[frame_idx] = []
                annotations[frame_idx].append((int(x), int(y), int(w), int(h), confidence, int(obj_id)))
        return annotations

    # def process_frame(self, frame, frame_idx, *args, **kwargs):
    #     """在视频帧中绘制标注，并统计 box 数量"""
    #     if frame_idx not in self.annotations or not self.show_annotations.get():
    #         return frame
    #
    #     # 初始化统计数据
    #     total_boxes = 0
    #     filtered_boxes = 0
    #
    #     # 遍历标注数据，绘制框并统计
    #     for x, y, w, h, confidence, obj_id in self.annotations[frame_idx]:
    #         total_boxes += 1  # 统计总的 box 数量
    #
    #         if confidence < self.confidence_threshold != 0.0:
    #             continue  # 跳过低于置信度阈值的标注
    #
    #         filtered_boxes += 1  # 统计通过置信度过滤的 box 数量
    #
    #         # 获取颜色
    #         color = self._get_color_for_id(obj_id)
    #
    #         # 如果没有 ID，使用默认颜色
    #         if obj_id == -1:
    #             color = (255, 255, 255)  # 白色
    #
    #         # 绘制矩形框
    #         frame = cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    #
    #         # 绘制标签
    #         label = f"ID: {obj_id}, Conf: {confidence:.2f}"
    #         frame = cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    #
    #         # 记录 ID 的历史位置
    #         if obj_id not in self.id_history:
    #             self.id_history[obj_id] = []
    #         self.id_history[obj_id].append((x + w // 2, y + h // 2))  # 记录中心点
    #
    #         # 绘制轨迹
    #         if self.tracking_enabled:
    #             for i in range(1, len(self.id_history[obj_id])):
    #                 cv2.line(frame, self.id_history[obj_id][i - 1], self.id_history[obj_id][i], color, 2)
    #
    #     self.stats[frame_idx] = {
    #         "total_boxes": total_boxes,
    #         "filtered_boxes": filtered_boxes,
    #     }
    #
    #     self._update_statistics(total_boxes, filtered_boxes)
    #     return self._draw_statistics_overlay(frame, total_boxes, filtered_boxes, frame_idx)

    def process_frame(self, frame, frame_idx, *args, **kwargs):
        """在视频帧中绘制标注，并统计 box 数量"""
        if frame_idx not in self.annotations or not self.show_annotations.get():
            return frame

        # 清理历史轨迹，保留最近的 N 帧
        max_history_length = 100  # 设置要保留的最大历史帧数
        for obj_id in list(self.id_history.keys()):
            if len(self.id_history[obj_id]) > max_history_length:
                self.id_history[obj_id] = self.id_history[obj_id][-max_history_length:]

        # 初始化统计数据
        total_boxes = 0
        filtered_boxes = 0

        # 遍历标注数据，绘制框并统计
        for x, y, w, h, confidence, obj_id in self.annotations[frame_idx]:
            total_boxes += 1  # 统计总的 box 数量

            if confidence < self.confidence_threshold != 0.0:
                continue  # 跳过低于置信度阈值的标注

            filtered_boxes += 1  # 统计通过置信度过滤的 box 数量

            # 获取颜色
            color = self._get_color_for_id(obj_id)

            # 如果没有 ID，使用默认颜色
            if obj_id == -1:
                color = (255, 255, 255)  # 白色

            # 绘制矩形框
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            # 绘制标签
            label = f"ID: {obj_id}, Conf: {confidence:.2f}"
            frame = cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

            # 记录 ID 的历史位置
            if obj_id not in self.id_history:
                self.id_history[obj_id] = []
            self.id_history[obj_id].append((x + w // 2, y + h // 2))  # 记录中心点

            # 绘制轨迹
            if self.tracking_enabled:
                for i in range(1, len(self.id_history[obj_id])):
                    cv2.line(frame, self.id_history[obj_id][i - 1], self.id_history[obj_id][i], color, 2)

        self.stats[frame_idx] = {
            "total_boxes": total_boxes,
            "filtered_boxes": filtered_boxes,
        }

        self._update_statistics(total_boxes, filtered_boxes)
        return self._draw_statistics_overlay(frame, total_boxes, filtered_boxes, frame_idx)
    def _update_statistics(self, total_boxes, filtered_boxes, max_history=100):
        """更新历史统计数据"""
        self.total_boxes_history.append(total_boxes)
        self.filtered_boxes_history.append(filtered_boxes)
        if len(self.total_boxes_history) > max_history:
            self.total_boxes_history.pop(0)
        if len(self.filtered_boxes_history) > max_history:
            self.filtered_boxes_history.pop(0)

    def resize_with_wheel(self, event):
        """通过鼠标滚轮调整信息统计面板大小和字体大小（对称缩放）"""
        scale_factor = 1.1  # 缩放因子，放大/缩小为 10%
        if event.delta > 0:  # 放大
            self.overlay_width = int(self.overlay_width * scale_factor)
            self.overlay_height = int(self.overlay_height * scale_factor)
            self.overlay_font_size = round(self.overlay_font_size * scale_factor, 2)
        else:  # 缩小
            self.overlay_width = max(100, int(self.overlay_width / scale_factor))
            self.overlay_height = max(50, int(self.overlay_height / scale_factor))
            self.overlay_font_size = max(0.1, round(self.overlay_font_size / scale_factor, 2))

        # 更新字体位置（文字相对于面板的比例保持一致）
        self.label_offset_x = int(0.05 * self.overlay_width)  # 距离面板左侧的偏移
        self.label_offset_y = int(0.1 * self.overlay_height)  # 距离面板顶部的偏移
        self.label_spacing = int(0.2 * self.overlay_height)  # 每行文字的间距

    def _draw_statistics_overlay(self, frame, total_boxes, filtered_boxes, current_frame_idx):
        overlay_x, overlay_y = 10, 90
        overlay_color = (50, 50, 50)
        alpha = 0.6

        # 创建 overlay
        overlay = frame.copy()

        # 绘制第一个统计信息面板
        cv2.rectangle(overlay, (overlay_x, overlay_y),
                      (overlay_x + self.overlay_width, overlay_y + self.overlay_height),
                      overlay_color, -1)

        # 计算第二个统计曲线面板的位置
        new_panel_y = overlay_y + self.overlay_height + 10

        # 绘制第二个统计曲线面板
        cv2.rectangle(overlay, (overlay_x, new_panel_y),
                      (overlay_x + self.overlay_width, new_panel_y + self.overlay_height),
                      overlay_color, -1)

        # 叠加透明效果
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # 计算第二个面板中字体的位置（面板底部）
        label_x = overlay_x + int(0.05 * self.overlay_width)  # 距离面板左侧的偏移量
        label_y_bottom = new_panel_y + self.overlay_height - int(0.05 * self.overlay_height)  # 距离面板底部的偏移量

        frame_text = f"Frame: {current_frame_idx}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_color = (255, 255, 255)  # 白色
        thickness = 2
        position = (10, 30)
        cv2.putText(frame, frame_text, position, font, font_scale, font_color, thickness, cv2.LINE_AA)

        # 在帧上显示置信度阈值
        confidence_text = f"Confidence Threshold: {self.confidence_threshold:.2f}"
        position = (10, 60)
        cv2.putText(frame, confidence_text, position, font, font_scale, font_color, thickness, cv2.LINE_AA)

        # 绘制统计信息文字到第一个面板
        cv2.putText(frame, f"Total Boxes: {total_boxes}",
                    (label_x, overlay_y + int(0.2 * self.overlay_height)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    self.overlay_font_size,
                    (0, 0, 255), 1, cv2.LINE_AA)

        cv2.putText(frame, f"Filtered Boxes: {filtered_boxes}",
                    (label_x, overlay_y + int(0.4 * self.overlay_height)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    self.overlay_font_size,
                    (0, 255, 0), 1, cv2.LINE_AA)

        # 绘制第二个面板底部的文字
        cv2.putText(frame, "Statistics Curve Panel",
                    (label_x, label_y_bottom),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    self.overlay_font_size,
                    (255, 255, 255), 1, cv2.LINE_AA)

        # 绘制曲线到第二个面板
        self._draw_curve(
            frame,
            self.total_boxes_per_frame,
            (overlay_x, new_panel_y, self.overlay_width, self.overlay_height),
            color=(0, 0, 255),
            font=cv2.FONT_HERSHEY_SIMPLEX,
            font_scale=self.overlay_font_size,
            font_color=(0, 0, 255),
            thickness=2
        )
        self._draw_curve(
            frame,
            self.filtered_boxes_per_frame,
            (overlay_x, new_panel_y, self.overlay_width, self.overlay_height),
            color=(0, 255, 0),
            font=cv2.FONT_HERSHEY_SIMPLEX,
            font_scale=self.overlay_font_size,
            font_color=(0, 255, 0),
            thickness=2
        )

        # 在曲线面板中高亮当前帧的点
        if 0 <= current_frame_idx < len(self.total_boxes_per_frame):
            # 计算当前帧点的 x 坐标
            current_point_x = int(
                overlay_x + (current_frame_idx / len(self.total_boxes_per_frame)) * self.overlay_width
            )

            # 计算当前帧点的 y 坐标（总 box 曲线）
            if max(self.total_boxes_per_frame) > 0:
                current_point_y_total = int(
                    new_panel_y + self.overlay_height - (
                            self.total_boxes_per_frame[current_frame_idx] / max(self.total_boxes_per_frame)
                    ) * self.overlay_height
                )
                cv2.circle(frame, (current_point_x, current_point_y_total), 5, (0, 0, 255), -1)  # 红点表示总 box 数

            # 计算当前帧点的 y 坐标（过滤后的 box 曲线）
            if max(self.filtered_boxes_per_frame) > 0:
                current_point_y_filtered = int(
                    new_panel_y + self.overlay_height - (
                            self.filtered_boxes_per_frame[current_frame_idx] / max(self.filtered_boxes_per_frame)
                    ) * self.overlay_height
                )
                cv2.circle(frame, (current_point_x, current_point_y_filtered), 5, (0, 255, 0), -1)  # 绿点表示过滤后的 box 数

        return frame

    def _draw_curve(self, frame, data, region, color,
                    font, font_scale, font_color, thickness, label=None, label_position=None):
        """Draw a curve in the specified region"""
        x, y, width, height = region
        if not data:
            return
        max_value = max(max(data), 1)
        min_value = min(data)
        if max_value == min_value:
            scale_y = height / (max_value + 1)  # Avoid division by zero
        else:
            scale_y = height / (max_value - min_value)
        scale_x = width / len(data)
        points = [
            (int(x + i * scale_x), int(y + height - (value - min_value) * scale_y))
            for i, value in enumerate(data)
        ]
        # print("Data:", data)
        # print("Region:", region, "Max Value:", max_value, "Min Value:", min_value,
        #       "Scale X:", scale_x, "Scale Y:", scale_y)
        for i in range(1, len(points)):
            cv2.line(frame, points[i - 1], points[i], color, 1, cv2.LINE_AA)
        if label:
            cv2.putText(frame, label, label_position, font, font_scale, font_color, thickness, cv2.LINE_AA)
