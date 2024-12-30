class BaseModule:
    """Base module class, all function modules inherit from this class."""

    def __init__(self, root, name, priority=100):
        self.root = root
        self.priority = priority
        self.name = name
        self.priority = priority

    def create_controls(self, parent_frame):
        """Creating controls on the control panel"""
        pass

    def process_frame(self, frame, frame_idx, *args, **kwargs):
        """Processing video/image frames"""
        return frame
