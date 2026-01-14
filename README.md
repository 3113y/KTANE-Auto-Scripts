# DAMN - Keep Talking and Nobody Explodes 单人通关助手

一个基于 PyQt5 开发的《Keep Talking and Nobody Explodes》游戏单人辅助工具，提供各种炸弹模块的解题算法。

## ✨ 功能特性

### 🎮 炸弹模块求解器
- **线路模块** - 基于线的颜色和序列号的复杂决策算法
- **按钮模块** - 根据按钮颜色、文字、电池数和指示灯状态判断
- **四色方块模块** - 基于序列号特征的解题逻辑
- **记忆模块** - 5阶段记忆跟踪算法
- **迷宫模块** - 交互式网格选择，动态显示解题路径

### 🤖 YOLO 自动识别（开发中）
- 基于 YOLOv8 的游戏模块自动检测
- 自动截图和数据采集工具
- 数据标注和模型训练脚本
- 详见 [YOLO 数据集准备指南](README_YOLO.md)

## 🚀 快速开始

### 安装依赖
```bash
pip install PyQt5
```

### 运行主程序
```bash
cd Damn
python Damn.py
```

## 📦 项目结构

```
Damn/
├── Damn.py                 # 主程序入口
├── scripts/                # 模块求解器
│   ├── line_module.py      # 线路模块
│   ├── button_module.py    # 按钮模块
│   ├── four_color_block_module.py  # 四色方块
│   ├── memory_module.py    # 记忆模块
│   └── maze_module.py      # 迷宫模块
└── pictures/               # 迷宫模块图片资源

scripts/                    # YOLO 工具脚本
├── dataset_collector.py    # 截图采集工具（F8/F9）
├── yolo_editor.py         # 标注编辑器
├── prepare_yolo_dataset.py # 数据集准备
└── train_yolo.py          # 模型训练
```

## 📖 使用说明

### 炸弹模块求解
1. 启动程序后选择对应的模块类型
2. 按照提示输入游戏中的信息
3. 程序会给出解题步骤和答案
4. 迷宫模块会显示完整的解题路径图

### YOLO 自动识别
参考 [YOLO 数据集准备指南](README_YOLO.md) 了解：
- 如何采集训练数据
- 如何标注和整理数据集
- 如何训练和使用模型

## 🛠️ 技术栈

- **GUI框架**: PyQt5
- **计算机视觉**: YOLOv8, OpenCV
- **深度学习**: PyTorch
- **开发语言**: Python 3.9+

## 📝 开发说明

详细的架构说明和开发指南请参考 [.github/copilot-instructions.md](.github/copilot-instructions.md)

## 📄 许可证

本项目仅供学习交流使用

## 🙏 致谢

游戏规则基于官方《Keep Talking and Nobody Explodes》手册实现
