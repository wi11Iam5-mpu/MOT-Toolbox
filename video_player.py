import os
import time
import threading
import cv2
# import tkinter as tk
import customtkinter as ctk
from PIL import Image, ImageTk
from tkinter import filedialog


class VideoPlayer:
    # 从 __init__ 方法中移除 resolution 参数
    def __init__(self, root, canvas, fps=30, default_video_path=None, default_image_path=None):
        self.root = root
        self.canvas = canvas
        self.cap = None
        self.playing = False
        self.current_frame_idx = 0
        self.total_frames = 0
        self.fps = fps
        self.frame_interval = 1 / fps
        self.tk_image = None

        self.default_video_path = default_video_path
        self.default_image_path = default_image_path
        self.image_folder = None
        self.image_files = []
        # self.current_frame_idx = 0

        self.modules = []
        self.split_positions = {}
        self.progress = None
        self.was_playing = False

        self.image_folder = None
        self.image_files = []
        # current_frame_idx = 0

    def load_video(self, file_path):
        """加载视频文件，并调整窗口大小"""
        if not file_path or not os.path.isfile(file_path):
            raise ValueError(f"Invalid video file path: {file_path}")

        # 释放旧的视频资源
        if self.cap:
            self.cap.release()

        self.cap = cv2.VideoCapture(file_path)
        self.image_folder = None
        if not self.cap.isOpened():
            raise ValueError("Cannot open video file")

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.frame_interval = 1 / (self.fps + 1e-6)
        self.current_frame_idx = 0

        self.update_canvas_size()

        if self.progress:
            self.progress.configure(to=self.total_frames - 1)

    def load_image_folder(self, folder_path):
        """加载图片文件夹，并调整窗口大小"""
        self.image_folder = folder_path
        self.image_files = sorted(
            [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        )
        self.cap = None
        self.total_frames = len(self.image_files)
        self.current_frame_idx = 0

        # 获取第一张图片的分辨率
        if self.image_files:
            first_image_path = os.path.join(folder_path, self.image_files[0])
            first_image = cv2.imread(first_image_path)
            if first_image is not None:
                image_height, image_width, _ = first_image.shape
                self.update_canvas_size()
            else:
                raise ValueError(f"Cannot read the first image: {first_image_path}")

        if self.progress:
            self.progress.configure(to=self.total_frames - 1)

    def setup_progress_bar(self, parent_frame):
        """设置进度条"""
        label = ctk.CTkLabel(parent_frame, text="Progress Bar")
        label.pack(side=ctk.LEFT, padx=5)
        progress = ctk.CTkSlider(parent_frame, from_=0, to=100, command=self.update_frame)
        progress.pack(side=ctk.LEFT, fill=ctk.X, expand=True)
        self.set_progress_bar(progress)

    def process_frame(self, frame, frame_idx):
        """按模块顺序处理帧"""
        for module in self.modules:
            frame = module.process_frame(frame, frame_idx)
        return frame

    def _display_frame(self, frame):
        """Display frame content and adjust resolution."""
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        frame = cv2.resize(frame, (canvas_width, canvas_height))

        # Convert frame to ImageTk format
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        self.tk_image = ImageTk.PhotoImage(image)

        # Use a cached canvas image to avoid creating new images
        if not hasattr(self, "canvas_image"):
            self.canvas_image = self.canvas.create_image(0, 0, anchor="nw", image=self.tk_image)
        else:
            self.canvas.itemconfig(self.canvas_image, image=self.tk_image)

    def add_module(self, module):
        """Add a processing module."""
        self.modules.append(module)
        # 按优先级排序，priority 值越小优先级越高
        self.modules.sort(key=lambda mod: mod.priority)
        print(f"Module {module.name} added with priority {module.priority}")
        if hasattr(module, "split_position"):
            self.split_positions[module] = module.split_position

    def get_current_frame(self):
        """返回当前帧数据"""
        if self.cap and self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
            ret, frame = self.cap.read()
            if ret:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)  # 保持帧位置不变
                return frame
        elif self.image_folder:
            if 0 <= self.current_frame_idx < len(self.image_files):
                image_path = os.path.join(self.image_folder, self.image_files[self.current_frame_idx])
                frame = cv2.imread(image_path)
                return frame
        return None

    def update_split_positions(self, event=None, video_player=None):
        """更新分屏位置并立即刷新画面"""
        if video_player:  # 如果传入了 VideoPlayer 实例，手动刷新当前帧
            current_frame = video_player.get_current_frame()
            if current_frame is not None:
                processed_frame = self.process_frame(current_frame, video_player.current_frame_idx)
                video_player._display_frame(processed_frame)

    def play(self):
        """Start video playback."""
        if not self.cap and not self.image_folder:
            return
        if not self.playing:
            self.playing = True
            threading.Thread(target=self._play_loop, daemon=True).start()

    def _play_loop(self):
        """Internal playback loop."""
        while self.playing:
            # start_time = time.time()  # 开始时间
            try:
                frame = None
                if self.cap and self.cap.isOpened():
                    # Read the next video frame
                    ret, frame = self.cap.read()
                    if not ret or frame is None:
                        print("Playback reached the end of the video or failed to read frame.")
                        self.stop()  # Stop playback on failure
                        break
                elif self.image_folder:
                    # Read the next image in the folder
                    if self.current_frame_idx >= len(self.image_files):
                        print("Playback reached the end of the image folder.")
                        self.stop()  # Stop playback
                        break
                    image_path = os.path.join(self.image_folder, self.image_files[self.current_frame_idx])
                    frame = cv2.imread(image_path)
                    if frame is None:
                        print(f"Failed to read image: {image_path}")
                        self.stop()
                        break

                # Process the frame using the modules
                for module in self.modules:
                    frame = module.process_frame(frame, self.current_frame_idx)

                # Update the current frame index
                self.current_frame_idx += 1
                # Display the processed frame
                self.root.after(0, self._display_frame, frame)

                # Update the progress bar if it exists
                if self.progress:
                    self.progress.set(self.current_frame_idx)

                # Wait for the next frame interval
                # elapsed_time = time.time() - start_time
                # print(f"Elapsed time: {elapsed_time:.3f}")
                time.sleep(self.frame_interval)

            except Exception as e:
                print(f"Error in playback loop: {e}")
                self.stop()
                break

    def pause(self):
        """Pause playback."""
        self.playing = False

    def update_canvas_size(self):
        import inspect
        stack = inspect.stack()
        caller_frame = stack[1]
        caller_name = caller_frame.function
        print(f"Updating canvas size - Called by function: {caller_name}")

        """动态调整图片尺寸以适应 Canvas"""
        if not self.playing:  # 确保仅在暂停或无手动更新时调整大小
            """动态调整图片尺寸以适应 Canvas，仅在必要时调用"""
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()

            if self.cap:  # 如果是视频
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, canvas_width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, canvas_height)
            elif self.image_folder and self.image_files:
                image_path = os.path.join(self.image_folder, self.image_files[0])
                frame = cv2.imread(image_path)
                if frame is not None:
                    frame = cv2.resize(frame, (canvas_width, canvas_height))
                    self._display_frame(frame)  # 仅显示一次

    def stop(self):
        """Stop playback and reset."""
        self.playing = False
        if self.cap:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.current_frame_idx = 0
            # self._clear_canvas()
            if self.progress:
                self.progress.set(0)

        if self.image_folder:
            self.current_frame_idx = 0
            # self._clear_canvas()
            if self.progress:
                self.progress.set(0)

    def _clear_canvas(self):
        """Clear the canvas."""
        self.canvas.delete("all")

    def update_frame(self, value):
        """Update the video frame when dragging the progress bar."""
        self.current_frame_idx = int(float(value))

        if self.cap:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
            ret, frame = self.cap.read()
            if ret:
                for module in self.modules:
                    frame = module.process_frame(frame, self.current_frame_idx)
                self.root.after(0, self._display_frame, frame)

        elif self.image_folder:
            image_path = os.path.join(self.image_folder, self.image_files[self.current_frame_idx])
            frame = cv2.imread(image_path)
            for module in self.modules:
                frame = module.process_frame(frame, self.current_frame_idx)
            self.root.after(0, self._display_frame, frame)

    def set_progress_bar(self, progress):
        """Set the progress bar widget."""
        self.progress = progress
        self.progress.bind("<ButtonPress-1>", self._on_slider_press)
        self.progress.bind("<ButtonRelease-1>", self._on_slider_release)

    def _on_slider_press(self, event):
        """Pause when the user starts dragging the progress bar."""
        self.was_playing = self.playing
        if self.playing:
            self.pause()

    def _on_slider_release(self, event):
        """Resume playback after dragging the progress bar."""
        if self.was_playing:
            self.play()

    def create_controls(self, parent_frame):
        """Create video control buttons."""
        load_images_button = ctk.CTkButton(parent_frame, text="Load Images", command=self.load_image_folder_dialog, width=80)
        load_images_button.pack(side="left", padx=5)

        load_video_button = ctk.CTkButton(parent_frame, text="Load Video", command=self.load_video_dialog, width=80)
        load_video_button.pack(side="left", padx=5)

        play_button = ctk.CTkButton(parent_frame, text="Play", command=self.play, width=80)
        play_button.pack(side="left", padx=5)

        pause_button = ctk.CTkButton(parent_frame, text="Pause", command=self.pause, width=80)
        pause_button.pack(side="left", padx=5)

        stop_button = ctk.CTkButton(parent_frame, text="Stop", command=self.stop, width=80)
        stop_button.pack(side="left", padx=5)


    def load_video_dialog(self):
        """Open file dialog to load video."""
        file_path = filedialog.askopenfilename(initialdir=self.default_video_path,
                                               filetypes=[("Video Files", "*.mp4 *.avi *.mov")])
        if file_path:
            self.load_video(file_path)

    def load_image_folder_dialog(self):
        """Open directory dialog to load images."""
        folder_path = filedialog.askdirectory(initialdir=self.default_image_path)
        if folder_path:
            self.load_image_folder(folder_path)
