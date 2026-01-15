# KTANE-Auto-Script    Keep Talking and Nobody Explodes 自动解题系统

基于计算机视觉和深度学习的《Keep Talking and Nobody Explodes》游戏自动识别与解题系统。使用 YOLO11 实现炸弹模块的实时检测，结合游戏规则算法实现全自动通关。

##  项目简介

本项目旨在通过计算机视觉和深度学习技术，实现《Keep Talking and Nobody Explodes》游戏的自动化拆弹。系统分为两大功能模块：

###  YOLO 自动识别系统（主要功能 - 开发中）

使用 YOLO11 深度学习模型实现游戏界面的实时识别和自动解题：

- **实时模块检测** - 基于 YOLO11 的炸弹模块自动识别
- **多模块支持** - 线路、按钮、四色方块、记忆、迷宫等模块
- **迭代训练系统** - 自我进化的数据标注和模型训练流程
- **自动截图采集** - 热键控制的数据采集工具（F8/F9）
- **智能决策引擎** - 根据识别结果自动执行游戏规则算法

### 🎮 手动解题辅助工具（辅助功能）

提供 GUI 界面的手动解题辅助工具，帮助玩家学习游戏规则：

- **线路模块** - 基于线的颜色和序列号的决策算法
- **按钮模块** - 根据按钮颜色、文字、电池数和指示灯状态判断
- **四色方块模块** - 基于序列号特征的解题逻辑
- **记忆模块** - 5阶段记忆跟踪算法
- **迷宫模块** - 交互式网格选择，动态显示解题路径

## 🛠️ 技术栈

**核心技术**

- **目标检测**: YOLO11 (Ultralytics) - 炸弹模块实时识别
- **深度学习框架**: PyTorch 2.x - 模型训练和推理
- **计算机视觉**: OpenCV - 图像处理和特征提取
- **屏幕控制**: PyAutoGUI - 游戏交互和数据采集

**辅助工具**

- **GUI 框架**: PyQt5 - 手动解题辅助界面
- **热键监听**: keyboard - 截图采集控制
- **图像处理**: Pillow - 数据预处理

**开发环境**
- **Python**: 3.9+
- **包管理**: uv (推荐) / pip

## 📦 项目结构

```
KTANE-Auto-Scripts/
├── scripts/                      # 🤖 自动化系统（主要功能）
│   ├── auto_bomb_solver.py      # 主控制程序
│   ├── vision_system.py         # 视觉识别系统
│   ├── decision_maker.py        # 决策算法系统
│   ├── control_system.py        # 游戏控制系统
│   ├── dataset_collector.py     # 数据采集（F8/F9）
│   ├── train_yolo.py            # 模型训练
│   └── iterative_annotation_system.py  # 迭代标注系统
│
├── dataset/                      # 📊 数据集管理
│   ├── raw_images/              # 原始截图
│   ├── labels/                  # YOLO 标注文件
│   └── yolo/                    # 训练数据集
│
├── runs/                         # 🏃 训练记录
│   └── detect/train/weights/    # 模型权重
│
├── Damn/                         # 🎮 手动辅助工具（辅助功能）
│   ├── Damn.py                  # GUI 主程序
│   ├── scripts/                 # 各模块求解器
│   │   ├── line_module.py       # 线路模块
│   │   ├── button_module.py     # 按钮模块
│   │   ├── four_color_block_module.py  # 四色方块
│   │   ├── memory_module.py     # 记忆模块
│   │   └── maze_module.py       # 迷宫模块
│   └── pictures/                # 迷宫模块图片资源
│
└── docs/                         # 📚 文档
    ├── README_YOLO.md           # YOLO 训练指南
    ├── ANNOTATION_GUIDE.md      # 标注系统指南
    └── FILE_STRUCTURE.md        # 文件结构说明
```

## 📖 快速开始

### 安装依赖

```bash
# 使用 uv（推荐）
uv pip install -r requirements.txt

# 或使用 pip
pip install -r requirements.txt
```

### 使用手动辅助工具

```bash
cd Damn
python Damn.py
```

### YOLO 自动识别系统

详细使用说明请参考：
- [YOLO 训练指南](docs/README_YOLO.md) - 完整训练流程
- [迭代标注系统](docs/ANNOTATION_GUIDE.md) - 高效数据标注
- [模型使用指南](docs/MODEL_USAGE.md) - 推理和测试

## 📝 文档

- 📁 [文件结构说明](docs/FILE_STRUCTURE.md) - 项目目录组织和架构
- 🏷️ [标注系统指南](docs/ANNOTATION_GUIDE.md) - 迭代式自动标注流程
- 🎯 [YOLO训练指南](docs/README_YOLO.md) - 模型训练详细步骤
- ⚙️ [环境配置记录](docs/SETUP.md) - 依赖和环境设置
- 🔧 [开发架构指南](.github/copilot-instructions.md) - 核心架构说明

## 📄 许可证

本项目仅供学习交流使用

## 🙏 致谢

游戏规则基于官方《Keep Talking and Nobody Explodes》手册实现
