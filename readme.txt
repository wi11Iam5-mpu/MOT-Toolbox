project/
│
├── main.py              # 主程序入口
├── video_player.py      # 视频播放器模块
├── modules/             # 功能模块目录
│   ├── __init__.py      # 模块目录初始化文件
│   ├── base_module.py   # 基础模块类
│   ├── video_overlay.py # 视频分屏模块
│   ├── video_annotation.py # 视频标注模块
│
├── config.json          # 配置文件
└── annotations/         # 示例标注文件目录
    └── example.txt

# TODO
0. 模块汇总页面
1. 集成行人轨迹生成
2. 集成伪平面估计
3. 集成斑马鱼轨迹生成
4. 集成轨迹段，CLIP生成
5. BUG: 其他动作更新，未清理原来的轨迹