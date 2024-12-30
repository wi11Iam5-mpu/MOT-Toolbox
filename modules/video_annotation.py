import cv2
import random
# import tkinter as tk
from tkinter import ttk, filedialog
import customtkinter as ctk

from .base_module import BaseModule


class VideoAnnotationModule(BaseModule):
    """Video annotation module: displays bounding boxes and labels for object detection in video frames"""

    def __init__(self, root, name='VideoAnnotationModule', priority=1, show_annotations=False,
                 default_annotation_path="", confidence_threshold=0.5):
        super().__init__(root, name, priority)
        self.show_annotations = ctk.BooleanVar(value=show_annotations)
        self.default_annotation_path = default_annotation_path
        self.confidence_threshold = confidence_threshold
        self.annotations = {}  # format：{frame_idx: [(x, y, w, h, confidence, id)]}
        self.tracking_enabled = False  # whether to enable track display
        self.id_history = {}  # store history IDs to plot trajectories

        self.stats = {}  # used to store statistics for each frame
        self.id_colors = {}  # used to store the color of each id

        # curve data
        self.total_boxes_history = []  # record total number of boxes
        self.filtered_boxes_history = []  # record the number of boxes filtered

        # size of statistical information panel
        self.overlay_width = 300
        self.overlay_height = 150
        self.overlay_font_size = 0.6
        self.resizing = False

        # Initialize font position and line spacing
        self.label_offset_x = int(0.05 * self.overlay_width)  # Offset from the left side of the panel
        self.label_offset_y = int(0.0 * self.overlay_height)  # Offset from the top of the panel
        self.label_spacing = int(0.2 * self.overlay_height)  # Spacing of text per line

        # Bind mouse events for resizing
        self.root.bind("<MouseWheel>", self.resize_with_wheel)

    def _get_color_for_id(self, obj_id):
        """Assign a unique color to each ID"""
        if obj_id not in self.id_colors:
            self.id_colors[obj_id] = (
                random.randint(0, 255),  # R
                random.randint(0, 255),  # G
                random.randint(0, 255),  # B
            )
        return self.id_colors[obj_id]

    def create_controls(self, parent_frame):
        """Creating Button and Slider Controls on a Control Panel"""
        load_button = ctk.CTkButton(parent_frame, text="Load Annotations", command=self.load_annotations)
        load_button.pack(side=ctk.LEFT, padx=5)

        toggle_checkbox = ctk.CTkCheckBox(parent_frame, text="Show", variable=self.show_annotations)
        toggle_checkbox.pack(side=ctk.LEFT, padx=5)

        # Slider for adjusting confidence thresholds
        label = ctk.CTkLabel(parent_frame, text="Confidence Threshold")
        label.pack(side=ctk.LEFT, padx=5)

        self.slider = ctk.CTkSlider(parent_frame, from_=0.0, to=1.0, number_of_steps=100)
        self.slider.set(self.confidence_threshold)
        self.slider.pack(side=ctk.LEFT, padx=5)
        self.slider.bind("<Motion>", self.update_confidence_threshold)

    def update_confidence_threshold(self, event=None, video_player=None):
        """Updates the confidence threshold and immediately refreshes the screen"""
        self.confidence_threshold = self.slider.get()
        if video_player:  # 如果传入了 VideoPlayer 实例，手动刷新当前帧
            current_frame = video_player.get_current_frame()
            if current_frame is not None:
                processed_frame = self.process_frame(current_frame, video_player.current_frame_idx)
                video_player._display_frame(processed_frame)
                self._precompute_statistics()  # 预计算统计数据

    def load_annotations(self):
        """Loading labeled files"""
        annotation_path = filedialog.askopenfilename(initialdir=self.default_annotation_path,
                                                     filetypes=[("Text Files", "*.txt")])
        if annotation_path:
            self.annotations = self._parse_annotations(annotation_path)
            self._check_tracking()  # 检查是否启用轨迹显示
            self._precompute_statistics()  # 预计算统计数据

    def _precompute_statistics(self):
        """Pre-calculated per-frame statistics to reduce real-time computation overheads"""
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
        """Check if track display is enabled"""
        for frame_idx, boxes in self.annotations.items():
            ids = [box[5] for box in boxes]  # 获取当前帧的所有 ID
            if 1 == len(set(ids)) or len(set(ids)) != len(ids):  # 检查 ID 是否有重复
                self.tracking_enabled = False
                break
        else:
            self.tracking_enabled = True

    def _parse_annotations(self, file_path):
        """Parsing annotated documents"""
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

    def process_frame(self, frame, frame_idx, *args, **kwargs):
        """Drawing annotations in video frames and counting the number of boxes"""
        if frame_idx not in self.annotations or not self.show_annotations.get():
            return frame

        # Cleans up the history track, keeping the most recent N frames
        max_history_length = 100
        for obj_id in list(self.id_history.keys()):
            if len(self.id_history[obj_id]) > max_history_length:
                self.id_history[obj_id] = self.id_history[obj_id][-max_history_length:]

        # Initialization statistics
        total_boxes = 0
        filtered_boxes = 0

        # 遍历标注数据，绘制框并统计
        for x, y, w, h, confidence, obj_id in self.annotations[frame_idx]:
            total_boxes += 1

            if confidence < self.confidence_threshold != 0.0:
                continue

            filtered_boxes += 1

            color = self._get_color_for_id(obj_id)

            # If no ID, use default color
            if obj_id == -1:
                color = (255, 255, 255)  # white

            # Drawing bounding-boxes
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            # Drawing labels
            label = f"ID: {obj_id}, Conf: {confidence:.2f}"
            frame = cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

            # Historical location of record ID
            if obj_id not in self.id_history:
                self.id_history[obj_id] = []
            self.id_history[obj_id].append((x + w // 2, y + h // 2))  # Record center point

            # Mapping trajectories
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
        """Updating historical statistics"""
        self.total_boxes_history.append(total_boxes)
        self.filtered_boxes_history.append(filtered_boxes)
        if len(self.total_boxes_history) > max_history:
            self.total_boxes_history.pop(0)
        if len(self.filtered_boxes_history) > max_history:
            self.filtered_boxes_history.pop(0)

    def resize_with_wheel(self, event):
        """Resize information statistics panel and font size via mouse wheel"""
        scale_factor = 1.1  # Scaling factor, zoom in/out as 10%
        if event.delta > 0:
            self.overlay_width = int(self.overlay_width * scale_factor)
            self.overlay_height = int(self.overlay_height * scale_factor)
            self.overlay_font_size = round(self.overlay_font_size * scale_factor, 2)
        else:
            self.overlay_width = max(100, int(self.overlay_width / scale_factor))
            self.overlay_height = max(50, int(self.overlay_height / scale_factor))
            self.overlay_font_size = max(0.1, round(self.overlay_font_size / scale_factor, 2))

        #  Update font position
        self.label_offset_x = int(0.05 * self.overlay_width)
        self.label_offset_y = int(0.1 * self.overlay_height)
        self.label_spacing = int(0.2 * self.overlay_height)

    def _draw_statistics_overlay(self, frame, total_boxes, filtered_boxes, current_frame_idx):
        overlay_x, overlay_y = 10, 90
        overlay_color = (50, 50, 50)
        alpha = 0.6

        # create overlay
        overlay = frame.copy()

        # Draw the first statistical information panel
        cv2.rectangle(overlay, (overlay_x, overlay_y),
                      (overlay_x + self.overlay_width, overlay_y + self.overlay_height),
                      overlay_color, -1)

        # Calculate the position of the second statistics curve panel
        new_panel_y = overlay_y + self.overlay_height + 10

        # Plotting the second panel of statistical curves
        cv2.rectangle(overlay, (overlay_x, new_panel_y),
                      (overlay_x + self.overlay_width, new_panel_y + self.overlay_height),
                      overlay_color, -1)

        # Overlay Transparency
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # Calculate the position of the font in the second panel (bottom of the panel)
        label_x = overlay_x + int(0.05 * self.overlay_width)
        label_y_bottom = new_panel_y + self.overlay_height - int(0.05 * self.overlay_height)

        frame_text = f"Frame: {current_frame_idx}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_color = (255, 255, 255)
        thickness = 2
        position = (10, 30)
        cv2.putText(frame, frame_text, position, font, font_scale, font_color, thickness, cv2.LINE_AA)

        # Display confidence thresholds on frames
        confidence_text = f"Confidence Threshold: {self.confidence_threshold:.2f}"
        position = (10, 60)
        cv2.putText(frame, confidence_text, position, font, font_scale, font_color, thickness, cv2.LINE_AA)

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

        cv2.putText(frame, "Statistics Curve Panel",
                    (label_x, label_y_bottom),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    self.overlay_font_size,
                    (255, 255, 255), 1, cv2.LINE_AA)

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

        # Highlight the point of the current frame in the Curves panel
        if 0 <= current_frame_idx < len(self.total_boxes_per_frame):
            # Calculate the x-coordinate of the current frame point
            current_point_x = int(
                overlay_x + (current_frame_idx / len(self.total_boxes_per_frame)) * self.overlay_width
            )

            # Calculate the y-coordinate of the current frame point (total box curve)
            if max(self.total_boxes_per_frame) > 0:
                current_point_y_total = int(
                    new_panel_y + self.overlay_height - (
                            self.total_boxes_per_frame[current_frame_idx] / max(self.total_boxes_per_frame)
                    ) * self.overlay_height
                )
                cv2.circle(frame, (current_point_x, current_point_y_total), 5, (0, 0, 255), -1)  # 红点表示总 box 数

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
