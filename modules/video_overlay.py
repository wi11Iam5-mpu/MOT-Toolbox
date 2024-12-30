import cv2
import numpy as np
import tkinter as tk
# from tkinter import ttk
from .base_module import BaseModule
import os
import customtkinter as ctk


class VideoOverlayModule(BaseModule):
    """视频分屏模块：左侧彩色，右侧灰度或深度图"""

    def __init__(self, root, split_position=0.5, name='VideoOverlayModule', default_depth_map_path="", priority=2):
        super().__init__(root, name, priority)
        self.split_position = split_position
        self.default_depth_map_path = default_depth_map_path
        self.depth_images = []  # 存储深度图文件路径
        self.depth_image_folder = None  # 深度图文件夹路径

    def create_controls(self, parent_frame):
        """在控制面板上创建滑块控件"""
        # 添加加载深度图的按钮
        load_depth_button = ctk.CTkButton(parent_frame, text="Load Depth Images", command=self.load_depth_images)
        load_depth_button.pack(side=ctk.LEFT, padx=5)

        label = ctk.CTkLabel(parent_frame, text="Split Position")
        label.pack(side=ctk.LEFT, padx=5)

        # 滑块控制分屏位置
        self.slider = ctk.CTkSlider(parent_frame, from_=0.01, to=0.99, number_of_steps=100)
        self.slider.set(self.split_position)
        self.slider.pack(side=ctk.LEFT, padx=5)
        self.slider.bind("<Motion>", self.update_split_position)

    def load_depth_images(self):
        """加载深度图文件夹"""
        self.depth_image_folder = tk.filedialog.askdirectory(initialdir=self.default_depth_map_path,
                                                             title="Select Depth Image Folder")
        if self.depth_image_folder:
            self.depth_images = [f for f in os.listdir(self.depth_image_folder) if
                                 f.endswith('.png') or f.endswith('.jpg') or f.endswith('.jpeg')]
            self.depth_images.sort()  # 确保顺序一致

    def update_split_position(self, event=None, video_player=None):
        """更新分屏位置"""
        self.split_position = self.slider.get()
        if video_player:  # 如果传入了 VideoPlayer 实例，手动刷新当前帧
            current_frame = video_player.get_current_frame()
            if current_frame is not None:
                processed_frame = self.process_frame(current_frame, video_player.current_frame_idx)
                video_player._display_frame(processed_frame)

    def process_frame(self, frame, frame_idx, *args, **kwargs):
        height, width, _ = frame.shape
        split_position = int(width * self.split_position)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_gray_3channel = cv2.merge((frame_gray, frame_gray, frame_gray))

        # 检查深度图是否可用
        if self.depth_image_folder and len(self.depth_images) > frame_idx:
            depth_image_path = os.path.join(self.depth_image_folder, self.depth_images[frame_idx])
            if os.path.isfile(depth_image_path):
                depth_frame = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)  # 读取深度图
                depth_frame = cv2.cvtColor(depth_frame, cv2.COLOR_BGR2RGB)
                # 将深度图叠加到右侧
                right_part = depth_frame[:, split_position:]
            else:
                right_part = frame_gray_3channel[:, split_position:]  # 默认使用灰度图
        else:
            right_part = frame_gray_3channel[:, split_position:]  # 默认使用灰度图

        left_part = frame_rgb[:, :split_position]
        combined_frame = np.hstack((left_part, right_part))

        # 确保返回 RGB 格式
        return cv2.cvtColor(combined_frame, cv2.COLOR_BGR2RGB)
