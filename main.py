import json
import customtkinter as ctk
from video_player import VideoPlayer
from modules.video_overlay import VideoOverlayModule
from modules.video_annotation import VideoAnnotationModule


class MainApplication:
    def __init__(self, root):
        self.root = root
        self.root.title("MOT Toolbox V0.4")

        # 设置主题
        ctk.set_appearance_mode("system")  # 可选 "light", "dark", "system"
        ctk.set_default_color_theme("dark-blue")  # 可选 "blue", "green", "dark-blue"

        # 使用 grid 布局保证画布和控件面板区域分离
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)  # 画布区域
        self.root.rowconfigure(1, weight=0)  # 控件面板区域

        # 创建画布区域
        self.canvas_frame = ctk.CTkFrame(self.root)
        self.canvas_frame.grid(row=0, column=0, sticky="nsew")  # 画布占据顶部区域

        self.canvas = ctk.CTkCanvas(self.canvas_frame, bg="black", width=640, height=360)  # 暂时保留 Canvas
        self.canvas.pack(fill=ctk.BOTH, expand=True)

        # 创建控件区域
        self.control_panel = ctk.CTkFrame(self.root, height=100)  # 设置固定高度
        self.control_panel.grid(row=1, column=0, sticky="ew")  # 控件固定在底部
        self.control_panel.grid_propagate(False)  # 禁止控件面板动态调整大小

        # 加载配置文件
        with open("config.json", "r") as f:
            config = json.load(f)

        # 初始化 VideoPlayer
        video_player_config = config.get("video_player", {})
        gui_config = config.get("gui", {})
        self.video_player = VideoPlayer(
            root=self.root,
            canvas=self.canvas,
            fps=video_player_config.get("fps", 30),
            default_video_path=video_player_config.get("default_video_path"),
            default_image_path=video_player_config.get("default_image_path"),
        )

        # 加载模块
        self.modules = []
        self.load_modules(config.get("modules", []))

        # 创建全局控制区域
        self.create_global_controls()

        # 根据配置文件设置窗口大小
        resolution = gui_config.get("resolution")  # Resolution from GUI config
        if resolution:
            width, height = resolution
            # 获取所有子面板的总高度
            control_panel_height = self.get_total_control_panel_height()
            total_height = height + control_panel_height
            self.root.geometry(f"{width}x{total_height}")

        # 加载快捷键
        self.shortcuts = config.get("shortcuts", {})
        self.bind_shortcuts()

    def get_total_control_panel_height(self):
        """递归计算所有子面板的总高度"""
        self.root.update_idletasks()  # 确保布局已完成
        total_height = 0
        for child in self.control_panel.winfo_children():  # 遍历所有子控件
            total_height += child.winfo_reqheight()  # 累加每个子控件的高度
        return total_height

    def load_modules(self, module_configs):
        """加载模块，并为每个模块创建单独的控件区域"""
        for module_config in module_configs:
            module_name = module_config.get("name")
            module_params = module_config.get("params", {})

            # 创建一个分区框架
            module_frame = ctk.CTkFrame(self.control_panel)
            module_frame.pack(side=ctk.TOP, fill=ctk.X, padx=5, pady=5)

            if module_name == "VideoOverlayModule":
                module = VideoOverlayModule(self.root, **module_params)
                module.create_controls(module_frame)

                # 使用闭包绑定事件，传递当前模块
                def bind_update_split_position(mod):
                    return lambda event: mod.update_split_position(event, self.video_player)

                module.slider.bind("<B1-Motion>", bind_update_split_position(module))

            elif module_name == "VideoAnnotationModule":
                module = VideoAnnotationModule(self.root, **module_params)
                module.create_controls(module_frame)

                # 使用闭包绑定事件，传递当前模块
                def bind_update_confidence_threshold(mod):
                    return lambda event: mod.update_confidence_threshold(event, self.video_player)

                module.slider.bind("<B1-Motion>", bind_update_confidence_threshold(module))
            else:
                continue  # Skip unknown modules

            self.modules.append(module)
            self.video_player.add_module(module)

            # 暂停播放时调整滑块
            for slider in [module.slider]:
                slider.bind("<ButtonPress-1>", lambda event: self.video_player.pause())
                slider.bind("<ButtonRelease-1>", lambda event: self.video_player.play())

    def create_global_controls(self):
        """创建全局视频控制区域"""
        global_controls = ctk.CTkFrame(self.control_panel)
        global_controls.pack(side=ctk.TOP, fill=ctk.X, padx=5, pady=5)

        self.video_player.create_controls(global_controls)
        self.video_player.setup_progress_bar(global_controls)

    def bind_shortcuts(self):
        """绑定快捷键操作"""
        for action, key in self.shortcuts.items():
            if action == "Play":
                self.root.bind(key, lambda event: self.video_player.play())
            elif action == "Pause":
                self.root.bind(key, lambda event: self.video_player.pause())
            elif action == "Stop":
                self.root.bind(key, lambda event: self.video_player.stop())
            elif action == "LoadVideo":
                self.root.bind(key, lambda event: self.video_player.load_video_dialog())
            elif action == "ToggleAnnotations":
                for module in self.modules:
                    if isinstance(module, VideoAnnotationModule):
                        self.root.bind(key, lambda event: module.toggle_annotations())


if __name__ == "__main__":
    root = ctk.CTk()
    app = MainApplication(root)
    root.mainloop()
