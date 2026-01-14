# 项目环境配置完成

## ✅ 已完成配置

### 1. Python 版本
- **Python 3.12.11** （使用 uv 管理）
- 位置：`.python-version` 文件已更新

### 2. 虚拟环境
- 路径：`.venv/`
- 创建方式：`uv venv --python 3.12`
- 激活命令：`.venv\Scripts\activate`

### 3. 已安装依赖

#### 核心应用依赖
- `pyqt5==5.15.11` - GUI 框架
- `pyqt5-qt5==5.15.2` - Qt 库（Windows 兼容版本）
- `pyqt5-sip==12.18.0` - Python/Qt 绑定

#### YOLO 数据采集依赖
- `pyautogui==0.9.54` - 屏幕截图
- `keyboard==0.13.5` - 热键监听
- `pillow==12.1.0` - 图像处理

#### 计算机视觉依赖
- `opencv-python==4.12.0.88` - 图像处理核心
- `numpy==2.2.6` - 数值计算

#### YOLO 训练依赖
- `ultralytics==8.4.0` - YOLOv8 框架
- `torch==2.9.1` - PyTorch 深度学习框架
- `torchvision==0.24.1` - 计算机视觉工具
- `pyyaml==6.0.3` - 配置文件管理

## 📋 使用说明

### 运行主程序
```bash
# 激活虚拟环境
.venv\Scripts\activate

# 运行程序
python Damn/Damn.py

# 或直接使用虚拟环境 Python
.venv\Scripts\python.exe Damn/Damn.py
```

### 数据采集流程
```bash
# 1. 采集游戏截图
python scripts/dataset_collector.py

# 2. （可选）半自动预标注
python scripts/auto_annotate_helper.py

# 3. 手动标注（使用 LabelImg）
labelImg dataset/raw_images dataset/labels

# 4. 整理数据集
python scripts/prepare_yolo_dataset.py

# 5. 训练 YOLO 模型
yolo detect train data=dataset/yolo/data.yaml model=yolov8n.pt epochs=100
```

### 管理依赖

#### 添加新依赖
```bash
uv pip install package-name
```

#### 查看已安装依赖
```bash
uv pip list
# 或
uv pip freeze
```

#### 导出依赖列表
```bash
uv pip freeze > requirements.txt
```

## 🔧 项目配置文件

### pyproject.toml
```toml
[project]
name = "damn"
version = "0.1.0"
description = "Keep Talking and Nobody Explodes 单人通关手册"
readme = "README.md"
requires-python = ">=3.12"
dependencies = [
    "pyqt5>=5.15.0",
    "opencv-python>=4.8.0",
    "numpy>=1.24.0",
    "pyautogui>=0.9.54",
    "keyboard>=0.13.5",
    "pillow>=10.0.0",
    "ultralytics>=8.0.0",
    "pyyaml>=6.0",
]

[tool.uv]
constraint-dependencies = [
    "pyqt5-qt5==5.15.2",  # Windows 兼容版本
]
```

## ⚠️ 注意事项

### Windows 平台特殊处理
- PyQt5 在 Windows 上需要使用特定版本的 `pyqt5-qt5==5.15.2`
- 已在配置文件中添加约束依赖

### UV 使用建议
- 使用 `uv pip` 而不是直接使用 `pip`
- UV 会自动管理虚拟环境和依赖解析
- 链接模式警告可以忽略，或设置 `UV_LINK_MODE=copy` 环境变量

### 文件结构
```
DAMN/
├── .venv/              # 虚拟环境（Python 3.12）
├── .python-version     # Python 版本固定文件
├── pyproject.toml      # 项目配置和依赖
├── Damn/
│   ├── Damn.py        # 主程序入口
│   ├── pictures/      # 资源文件
│   └── scripts/       # 模块实现
├── scripts/           # 数据采集脚本
│   ├── dataset_collector.py
│   ├── auto_annotate_helper.py
│   └── prepare_yolo_dataset.py
└── dataset/           # 数据集目录（运行后生成）
    ├── raw_images/
    ├── labels/
    └── yolo/
```

## 🎯 下一步

1. ✅ 环境配置完成
2. 📸 开始采集游戏截图数据
3. 🏷️ 标注数据集
4. 🚀 训练 YOLO 模型
5. 🔗 集成模型到主程序

详细步骤请参考：
- [README_YOLO.md](README_YOLO.md) - YOLO 训练完整指南
- [.github/copilot-instructions.md](.github/copilot-instructions.md) - 项目架构文档
