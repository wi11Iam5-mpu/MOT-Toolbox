import json
import customtkinter as ctk
from video_player import VideoPlayer
from modules.video_overlay import VideoOverlayModule
from modules.video_annotation import VideoAnnotationModule


class MainApplication:
    def __init__(self, root):
        self.root = root
        self.root.title("MOT Toolbox V0.4 by wi11iam5")

        # Setting up our theme
        ctk.set_appearance_mode("system")  # optional "light", "dark", "system"
        ctk.set_default_color_theme("dark-blue")  # optional "blue", "green", "dark-blue"

        # Use a grid layout for the root window
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)  # canvas area
        self.root.rowconfigure(1, weight=0)  # control panel area

        # create a canvas area
        self.canvas_frame = ctk.CTkFrame(self.root)
        self.canvas_frame.grid(row=0, column=0, sticky="nsew")

        self.canvas = ctk.CTkCanvas(self.canvas_frame, bg="black", width=640, height=360)
        self.canvas.pack(fill=ctk.BOTH, expand=True)

        # create a control panel area
        self.control_panel = ctk.CTkFrame(self.root, height=100)
        self.control_panel.grid(row=1, column=0, sticky="ew")
        self.control_panel.grid_propagate(False)  # disable dynamic resizing of control panels

        # load configuration file
        with open("config.json", "r") as f:
            config = json.load(f)

        # init VideoPlayer
        video_player_config = config.get("video_player", {})
        gui_config = config.get("gui", {})
        self.video_player = VideoPlayer(
            root=self.root,
            canvas=self.canvas,
            fps=video_player_config.get("fps", 30),
            default_video_path=video_player_config.get("default_video_path"),
            default_image_path=video_player_config.get("default_image_path"),
        )

        # load modules
        self.modules = []
        self.load_modules(config.get("modules", []))

        # crete global controls
        self.create_global_controls()

        # set up GUI resolution
        resolution = gui_config.get("resolution")  # resolution from GUI config
        if resolution:
            width, height = resolution
            # get the total height of all sub-panels
            control_panel_height = self.get_total_control_panel_height()
            total_height = height + control_panel_height
            self.root.geometry(f"{width}x{total_height}")

        # load shortcuts
        self.shortcuts = config.get("shortcuts", {})
        self.bind_shortcuts()

    def get_total_control_panel_height(self):
        """Calculate the total height of all sub-panels"""
        self.root.update_idletasks()
        total_height = 0
        for child in self.control_panel.winfo_children():
            total_height += child.winfo_reqheight()
        return total_height

    def load_modules(self, module_configs):
        """Load the modules and create separate control areas for each module"""
        for module_config in module_configs:
            module_name = module_config.get("name")
            module_params = module_config.get("params", {})

            # Create a partitioned framework
            module_frame = ctk.CTkFrame(self.control_panel)
            module_frame.pack(side=ctk.TOP, fill=ctk.X, padx=5, pady=5)

            if module_name == "VideoOverlayModule":
                module = VideoOverlayModule(self.root, **module_params)
                module.create_controls(module_frame)

                # Bind events using closures, passing the current module
                def bind_update_split_position(mod):
                    return lambda event: mod.update_split_position(event, self.video_player)

                module.slider.bind("<B1-Motion>", bind_update_split_position(module))

            elif module_name == "VideoAnnotationModule":
                module = VideoAnnotationModule(self.root, **module_params)
                module.create_controls(module_frame)

                def bind_update_confidence_threshold(mod):
                    return lambda event: mod.update_confidence_threshold(event, self.video_player)

                module.slider.bind("<B1-Motion>", bind_update_confidence_threshold(module))
            else:
                continue  # Skip unknown modules

            self.modules.append(module)
            self.video_player.add_module(module)

            # adjust sliders while pausing playback
            for slider in [module.slider]:
                slider.bind("<ButtonPress-1>", lambda event: self.video_player.pause())
                slider.bind("<ButtonRelease-1>", lambda event: self.video_player.play())

    def create_global_controls(self):
        """Create global video control zones"""
        global_controls = ctk.CTkFrame(self.control_panel)
        global_controls.pack(side=ctk.TOP, fill=ctk.X, padx=5, pady=5)

        self.video_player.create_controls(global_controls)
        self.video_player.setup_progress_bar(global_controls)

    def bind_shortcuts(self):
        """Binding shortcut operation"""
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
