# YOLO 自动识别数据集准备指南

## 📋 总体流程

```
1. 采集数据 → 2. 标注数据 → 3. 整理数据集 → 4. 训练模型 → 5. 集成应用
```

## 🎯 第一步：采集游戏截图

### 使用自动截图工具

```bash
cd Damn
python scripts/dataset_collector.py
```

**操作说明：**
- 按 `F8` - 截取全屏
- 按 `F9` - 手动选择区域截图（推荐）
- 按 `ESC` - 退出程序

**采集建议：**
- 每个模块类型至少 50-100 张图片
- 包含不同的光照、角度、分辨率
- 包含不同的游戏状态（初始、进行中等）
- 截图保存在 `dataset/raw_images/` 目录

## 🏷️ 第二步：标注数据

### 方法1：使用LabelImg（推荐）

#### 安装 LabelImg
```bash
pip install labelImg
```

#### 启动标注工具
```bash
labelImg dataset/raw_images dataset/labels
```

#### 标注步骤
1. 点击 "Open Dir" 打开 `dataset/raw_images`
2. 点击 "Change Save Dir" 设置保存到 `dataset/labels`
3. 按 `W` 键开始框选目标
4. 选择类别（参考下面的类别定义）
5. 按 `Ctrl+S` 保存
6. 按 `D` 键切换到下一张图片

### 方法2：使用半自动标注辅助（可选）

先运行预标注脚本生成初步标注：

```bash
python scripts/auto_annotate_helper.py
```

然后用 LabelImg 检查和修正自动标注。

## 📦 类别定义

编号 | 类别名称 | 说明
-----|---------|-----
0 | line_module | 线路模块
1 | button_module | 按钮模块
2 | four_color_block | 四色方块模块
3 | memory_module | 记忆模块
4 | maze_module | 迷宫模块
5 | serial_number | 序列号区域
6 | battery_holder | 电池仓
7 | indicator_light | 指示灯

**注意：** 根据实际需求可以增减类别，修改 `prepare_yolo_dataset.py` 中的 `class_names` 列表。

## 🗂️ 第三步：整理数据集

标注完成后，运行数据集整理脚本：

```bash
python scripts/prepare_yolo_dataset.py
```

此脚本会：
- 自动划分训练集（80%）和验证集（20%）
- 创建YOLO标准目录结构
- 生成 `data.yaml` 配置文件

生成的目录结构：
```
dataset/yolo/
├── images/
│   ├── train/    # 训练集图片
│   └── val/      # 验证集图片
├── labels/
│   ├── train/    # 训练集标注
│   └── val/      # 验证集标注
└── data.yaml     # YOLO配置文件
```

## 🚀 第四步：训练YOLO模型

### 安装 Ultralytics YOLO

```bash
pip install ultralytics
```

### 开始训练

```bash
# 使用YOLOv8n（轻量级）
yolo detect train data=dataset/yolo/data.yaml model=yolov8n.pt epochs=100 imgsz=640

# 或使用YOLOv8s（更高精度）
yolo detect train data=dataset/yolo/data.yaml model=yolov8s.pt epochs=150 imgsz=640
```

**训练参数说明：**
- `data`: 数据集配置文件路径
- `model`: 预训练模型（n=nano, s=small, m=medium, l=large）
- `epochs`: 训练轮数（建议100-200）
- `imgsz`: 输入图片尺寸（640是标准尺寸）
- `batch`: 批大小（默认16，可根据显存调整）

### 训练进度监控

训练过程会在 `runs/detect/train/` 目录生成：
- `weights/best.pt` - 最佳模型权重
- `weights/last.pt` - 最后一次训练权重
- `results.png` - 训练曲线图
- `confusion_matrix.png` - 混淆矩阵

## 🧪 第五步：测试模型

### 单张图片测试

```bash
yolo detect predict model=runs/detect/train/weights/best.pt source=test_image.png
```

### Python代码测试

```python
from ultralytics import YOLO

# 加载模型
model = YOLO('runs/detect/train/weights/best.pt')

# 预测
results = model('test_image.png')

# 查看结果
for result in results:
    boxes = result.boxes  # 边界框
    for box in boxes:
        cls = int(box.cls[0])  # 类别
        conf = float(box.conf[0])  # 置信度
        xyxy = box.xyxy[0].tolist()  # 坐标
        print(f"类别: {cls}, 置信度: {conf:.2f}, 位置: {xyxy}")
```

## 🔧 常见问题

### Q: 需要多少数据？
A: 每个类别至少50张，推荐100-200张。数据质量比数量更重要。

### Q: 训练需要多久？
A: 取决于数据量和硬件。100 epochs通常需要：
- GPU (RTX 3060): 30分钟 - 2小时
- CPU: 数小时 - 十几小时

### Q: 如何提高精度？
A: 
1. 增加训练数据
2. 提高标注质量（框要准确）
3. 增加训练轮数
4. 使用更大的模型（yolov8s/m）
5. 数据增强（YOLO自动应用）

### Q: 标注文件格式？
A: YOLO格式（txt文件），每行：
```
class_id x_center y_center width height
```
所有值都是归一化到[0,1]的比例。

## 📚 参考资源

- [Ultralytics YOLO文档](https://docs.ultralytics.com/)
- [LabelImg GitHub](https://github.com/tzutalin/labelImg)
- [YOLO标注格式说明](https://docs.ultralytics.com/datasets/detect/)

## 🎓 下一步

训练完成后，参考主项目文档将YOLO模型集成到 `Damn.py` 中，实现自动识别功能。
