# 训练好的YOLO模型使用指南

训练完成后，模型权重保存在 `runs/detect/train/weights/` 目录下。本文档介绍如何使用这些模型。

## 📁 模型文件位置

训练完成后会生成以下文件：

```
runs/detect/train/
├── weights/
│   ├── best.pt      # 🌟 最佳模型（验证集上效果最好）
│   └── last.pt      # 最后一轮的模型
├── results.csv      # 训练指标
├── results.png      # 训练曲线图
└── confusion_matrix.png  # 混淆矩阵
```

**推荐使用 `best.pt`** - 这是验证集上表现最好的模型权重。

## 🚀 使用方式

### 方式一：快速测试脚本

使用提供的测试脚本快速验证模型效果：

```bash
python scripts/test_model.py
```

**功能选项：**

1. **实时屏幕检测** 
   - 自动截取主显示器画面
   - 实时显示检测结果
   - 按 ESC 退出
   - 适合：游戏中实时测试效果

2. **单张图片测试**
   - 在指定图片上进行推理
   - 保存结果到 `test_result.jpg`
   - 适合：精确检验某张图的效果

3. **批量测试数据集**
   - 测试整个文件夹的所有图片
   - 结果保存到 `runs/detect/test/`
   - 适合：评估模型整体性能

### 方式二：Python 代码集成

在你自己的代码中使用模型：

```python
from ultralytics import YOLO
import cv2

# 1. 加载模型
model = YOLO("runs/detect/train/weights/best.pt")

# 2. 推理
# 在图片文件上
results = model("game_screenshot.jpg")

# 在 OpenCV 图像数组上
img = cv2.imread("game_screenshot.jpg")
results = model(img, conf=0.5)  # conf: 置信度阈值

# 在摄像头上（实时）
results = model(source=0)  # 0 = 默认摄像头

# 3. 处理结果
for result in results:
    # 获取检测框
    boxes = result.boxes
    
    for box in boxes:
        # 边界框坐标 (x1, y1, x2, y2)
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        
        # 类别和置信度
        class_id = int(box.cls[0])
        class_name = result.names[class_id]
        confidence = float(box.conf[0])
        
        print(f"检测到: {class_name}, 置信度: {confidence:.2%}")
        print(f"位置: ({x1:.0f}, {y1:.0f}) -> ({x2:.0f}, {y2:.0f})")
    
    # 可视化（绘制边界框）
    annotated = result.plot()
    cv2.imshow("Result", annotated)
    cv2.waitKey(0)
```

### 方式三：使用 VisionSystem 类

项目提供了封装好的视觉系统类：

```python
from scripts.vision_system import VisionSystem

# 初始化（会自动加载模型）
vision = VisionSystem(model_path="runs/detect/train/weights/best.pt")

# 截屏并检测
img = vision.capture_screen()  # 自动截取主显示器
detections = vision.detect_objects(img, conf_threshold=0.5)

# 处理检测结果
for det in detections:
    print(f"{det.class_name}: {det.confidence:.2%}")
    print(f"中心点: {det.center}")
    print(f"边界框: {det.bbox}")

# 可视化
img_vis = vision.visualize_detections(img, detections)
cv2.imshow("Detections", img_vis)
cv2.waitKey(0)
```

### 方式四：集成到自动拆弹系统

完整的自动化流程：

```bash
python scripts/auto_bomb_solver.py
```

这会启动全自动拆弹系统，包含：
- 自动屏幕识别（使用你训练的模型）
- 智能决策算法
- 自动鼠标/键盘控制
- 模块解题逻辑

**注意：** 自动拆弹系统还在开发中，部分功能可能需要进一步实现。

## 🔧 模型参数调优

### 置信度阈值 (confidence threshold)

控制检测的严格程度：

```python
# 高置信度（更准确，但可能漏检）
results = model(img, conf=0.7)

# 中等置信度（平衡）
results = model(img, conf=0.5)

# 低置信度（检测更多，但误报增加）
results = model(img, conf=0.3)
```

**建议：**
- 游戏中实时使用：`conf=0.5`
- 数据标注辅助：`conf=0.3`
- 生产环境：`conf=0.6-0.7`

### IOU 阈值 (Non-Maximum Suppression)

控制重复框的过滤：

```python
results = model(img, conf=0.5, iou=0.45)  # 默认 0.45
```

### 推理设备

```python
# 自动选择（优先GPU）
results = model(img)

# 强制使用GPU
results = model(img, device=0)  # GPU 0

# 强制使用CPU
results = model(img, device='cpu')
```

## 📊 评估模型性能

### 查看训练结果

```bash
# 查看训练曲线
# 文件: runs/detect/train/results.png
# 指标: mAP50, mAP50-95, Precision, Recall, Loss

# 查看混淆矩阵
# 文件: runs/detect/train/confusion_matrix.png
# 显示各类别的预测准确性
```

### 在验证集上测试

```bash
python scripts/train_yolo.py
# 然后选择"验证模型"选项
```

### 自定义评估脚本

```python
from ultralytics import YOLO

model = YOLO("runs/detect/train/weights/best.pt")

# 在验证集上评估
metrics = model.val(data="dataset/yolo/data.yaml")

print(f"mAP50: {metrics.box.map50:.3f}")
print(f"mAP50-95: {metrics.box.map:.3f}")
print(f"Precision: {metrics.box.mp:.3f}")
print(f"Recall: {metrics.box.mr:.3f}")
```

## 🐛 常见问题

### 1. 模型加载失败

**问题：** `FileNotFoundError: runs/detect/train/weights/best.pt not found`

**解决：**
```python
from pathlib import Path

# 检查文件是否存在
model_path = Path("runs/detect/train/weights/best.pt")
if not model_path.exists():
    print("模型文件不存在，请先训练模型")
    print("运行: python scripts/train_yolo.py")
```

### 2. 检测效果不好

**可能原因：**
- 训练数据不足（需要更多标注数据）
- 训练轮数太少（增加 epochs）
- 类别不平衡（某些模块样本太少）
- 测试场景与训练数据差异大

**改进方法：**
1. 使用 `scripts/iterative_annotation_system.py` 迭代增强数据
2. 增加训练轮数：`epochs=200`
3. 调整数据增强参数
4. 收集更多样本（不同光照、角度、分辨率）

### 3. 推理速度慢

**优化方法：**
```python
# 使用半精度推理（需要GPU支持）
model = YOLO("best.pt")
model.to('cuda')  # 确保在GPU上
results = model(img, half=True)  # FP16 推理

# 或导出为更快的格式
model.export(format='onnx')  # 导出为 ONNX
model.export(format='tensorrt')  # 导出为 TensorRT (最快)
```

### 4. 多个模型如何选择

如果训练了多次（train, train2, train3...）：

```python
# 比较不同模型
import pandas as pd

# 读取训练结果
df1 = pd.read_csv("runs/detect/train/results.csv")
df2 = pd.read_csv("runs/detect/train2/results.csv")

# 比较最终 mAP
print("模型1 最佳 mAP50:", df1['metrics/mAP50(B)'].max())
print("模型2 最佳 mAP50:", df2['metrics/mAP50(B)'].max())

# 选择 mAP 更高的模型
```

## 📚 进阶用法

### 模型微调

如果已有模型，但想在新数据上继续训练：

```python
from ultralytics import YOLO

# 加载已训练的模型
model = YOLO("runs/detect/train/weights/best.pt")

# 在新数据上继续训练
model.train(
    data="dataset/yolo/data.yaml",
    epochs=50,  # 较少的轮数
    lr0=0.001,  # 较小的学习率
    resume=True  # 继续训练
)
```

### 模型集成（Ensemble）

使用多个模型提高准确性：

```python
from ultralytics import YOLO

# 加载多个模型
model1 = YOLO("runs/detect/train/weights/best.pt")
model2 = YOLO("runs/detect/train2/weights/best.pt")

# 分别推理
results1 = model1(img, conf=0.5)
results2 = model2(img, conf=0.5)

# 合并结果（需要自己实现NMS等）
# ...
```

### 导出为不同格式

```python
model = YOLO("runs/detect/train/weights/best.pt")

# ONNX（跨平台）
model.export(format='onnx')

# TensorRT（NVIDIA GPU 最快）
model.export(format='engine')

# CoreML（Apple 设备）
model.export(format='coreml')

# TFLite（移动端）
model.export(format='tflite')
```

## 🎯 最佳实践

1. **始终使用 best.pt** - 不是 last.pt
2. **先在测试图片上验证** - 再用于实时场景
3. **根据场景调整置信度** - 不要一刀切
4. **监控 GPU 内存** - 批量处理时注意内存溢出
5. **定期收集难例** - 持续改进模型

## 🔗 相关文档

- [训练 YOLO 模型](README_YOLO.md)
- [数据标注指南](ANNOTATION_GUIDE.md)
- [项目结构说明](FILE_STRUCTURE.md)
- [迭代训练系统](../scripts/iterative_annotation_system.py)

---

**遇到问题？** 检查模型训练日志：`runs/detect/train/results.csv`
