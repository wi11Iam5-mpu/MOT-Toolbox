class BaseModule:
    """基础模块类，所有功能模块继承此类"""

    def __init__(self, root, name, priority=100):
        self.root = root
        self.priority = priority
        self.name = name
        self.priority = priority

    def create_controls(self, parent_frame):
        """创建控制面板上的控件"""
        pass

    def process_frame(self, frame, frame_idx, *args, **kwargs):
        """处理视频帧"""
        return frame
