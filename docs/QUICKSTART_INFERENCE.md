# 🎯 快速开始 - 使用训练好的模型

训练完成后，3分钟内快速测试你的模型！

## ✨ 新功能

- **窗口捕获** - 只捕获游戏窗口，不受其他程序干扰
- **图像压缩** - 自动压缩到960px再识别，速度提升3-5倍！
- **OBS支持** - 可使用OBS虚拟摄像头作为输入源

## Step 1: 确认模型文件 ✅

检查训练是否成功：

```bash
# 查看模型文件
dir runs\detect\train\weights

# 应该看到：
# best.pt  <- 使用这个！
# last.pt
```

如果没有这个文件夹，说明还没训练模型，请运行：
```bash
python scripts/train_yolo.py
```

## Step 2: 快速测试 🚀

### 方法 A：实时检测（最直观）⭐ 推荐

```bash
python scripts/test_model.py
```

**选择模式1，然后选择捕获源：**

1. **全屏捕获** - 捕获整个主显示器
2. **窗口捕获** - 🎯 **推荐！** 只捕获游戏窗口
   - 自动识别可能的游戏窗口
   - 或手动输入窗口标题关键词（如"Keep Talking"）
   - 不受其他程序干扰
3. **OBS虚拟摄像头** - 使用OBS的输出
   - 适合推流或录制场景
   - 需要先在OBS中启动虚拟摄像头

**配置选项：**
- **目标尺寸**：960（默认）- 图像压缩到960px再识别
  - 960px：速度快，精度足够
  - 1280px：更高精度，速度稍慢
  - 640px：最快，精度可能下降
- **置信度阈值**：0.5（默认）
  - 0.3：检测更多目标，可能误报
  - 0.7：更准确，可能漏检

**效果：**
- ✅ 绿色框 + 类别名称 + 置信度
- ✅ 显示检测数量和FPS
- ✅ 显示原始尺寸和推理尺寸
- ✅ 按 ESC 退出

**试试这样：**
- 打开游戏窗口
- 运行测试脚本，选择"窗口捕获"
- 看看能识别出哪些模块！

### 方法 B：测试单张图片

如果有保存的游戏截图：

```bash
python scripts/test_model.py
# 输入 2
# 然后输入图片路径，例如：dataset/raw_images/screen_xxx.jpg
```

### 方法 C：在 Python 中使用

创建 `my_test.py`：

```python
from ultralytics import YOLO
import cv2

# 加载模型
model = YOLO("runs/detect/train/weights/best.pt")

# 测试图片
results = model("你的截图.jpg")

# 显示结果
for r in results:
    r.show()  # 弹出窗口
    
    # 打印检测到什么
    for box in r.boxes:
        name = r.names[int(box.cls[0])]
        conf = float(box.conf[0])
        print(f"检测到: {name}, 置信度: {conf:.0%}")
```

运行：
```bash
python my_test.py
```

## Step 3: 评估效果 📊

### 查看训练结果

```bash
# 训练曲线图
start runs\detect\train\results.png

# 混淆矩阵（看看哪些类容易混淆）
start runs\detect\train\confusion_matrix.png
```

### 关键指标解读

打开 `runs/detect/train/results.csv`，查看最后一行：

- **mAP50** > 0.8 ✅ 很好
- **mAP50** 0.5-0.8 ⚠️ 还行，可能需要更多数据
- **mAP50** < 0.5 ❌ 需要改进（更多数据或更长训练）

## Step 4: 集成到项目 🔧

### 使用 VisionSystem 类（高级）

项目提供了封装好的视觉系统类，支持窗口捕获和图像压缩：

```python
from scripts.vision_system import VisionSystem
import cv2

# 方式1：捕获整个屏幕
vision = VisionSystem(
    model_path="runs/detect/train/weights/best.pt",
    target_size=960  # 压缩到960px
)

# 方式2：只捕获游戏窗口（推荐）
vision = VisionSystem(
    model_path="runs/detect/train/weights/best.pt",
    target_size=960,
    window_title="Keep Talking"  # 窗口标题关键词
)

# 截屏并检测
img = vision.capture_screen()  # 自动截取（窗口或屏幕）
detections = vision.detect_objects(img, conf_threshold=0.5)

# 查看结果
for det in detections:
    print(f"{det.class_name}: {det.confidence:.0%} at {det.center}")

# 可视化
img_vis = vision.visualize_detections(img, detections)
cv2.imshow("Result", img_vis)
cv2.waitKey(0)
```

**窗口捕获的优势：**
- ✅ 只捕获游戏画面，不受其他窗口干扰
- ✅ 即使游戏窗口被部分遮挡也能正常工作
- ✅ 支持最小化后台捕获（某些游戏）

**图像压缩的好处：**
- ✅ 推理速度提升 3-5 倍
- ✅ GPU显存占用减少
- ✅ 检测框自动还原到原始尺寸
- ✅ 精度几乎无损失（960px对游戏足够）

### 启动自动拆弹系统

```bash
python scripts/auto_bomb_solver.py
```

（注意：完整的自动化功能还在开发中）

## 常见问题 ❓

### Q1: 模型检测不到东西？

**可能原因：**
1. 置信度阈值太高
   ```python
   # 试试降低阈值
   results = model(img, conf=0.3)  # 从 0.5 降到 0.3
   ```

2. 图像压缩太小
   ```bash
   # 运行时选择更大的尺寸
   # 目标尺寸：1280  # 而不是960
   ```

3. 训练数据不足
   - 需要更多标注样本
   - 运行：`python scripts/iterative_annotation_system.py`

4. 测试图片和训练数据差异大
   - 分辨率不同
   - 光照条件不同
   - 游戏版本不同

### Q2: 检测很多误报？

```python
# 提高置信度阈值
results = model(img, conf=0.7)  # 从 0.5 提高到 0.7
```

### Q3: 推理速度慢？

**优化方案（按优先级）：**

1. **使用图像压缩**（最有效）
   ```bash
   # 运行时选择
   目标尺寸：640  # 最快，或
   目标尺寸：960  # 平衡
   ```

2. **确保使用GPU**
   ```python
   import torch
   print(torch.cuda.is_available())  # 应该是 True
   ```

3. **使用窗口捕获代替全屏**
   - 窗口通常比全屏小，捕获更快

4. **如果仍然慢，安装GPU版PyTorch**
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

### Q4: 窗口捕获失败？

**解决方案：**
1. 确保游戏窗口可见（不要最小化）
2. 手动输入准确的窗口标题关键词
3. 使用管理员权限运行脚本（某些游戏需要）
4. 如果还是失败，切换到全屏捕获

### Q5: OBS虚拟摄像头无法使用？

**检查步骤：**
1. 确保OBS已安装并启动
2. 在OBS中点击"启动虚拟摄像头"
3. 测试脚本中选择正确的摄像头编号（通常是0或1）
4. 如果找不到，尝试其他编号（0-4）

### Q4: 想继续改进模型？

```bash
# 1. 收集更多数据
python scripts/dataset_collector.py

# 2. 使用迭代标注系统
python scripts/iterative_annotation_system.py

# 3. 重新训练（会自动创建 train2, train3...）
python scripts/train_yolo.py
```

## 下一步 🎓

✅ **基础使用** - 你已经会了！  
📚 **深入学习** - 阅读 [MODEL_USAGE.md](MODEL_USAGE.md)  
🔄 **持续改进** - 使用 [迭代训练系统](../scripts/iterative_annotation_system.py)  
🤖 **全自动化** - 完善 [自动拆弹系统](../scripts/auto_bomb_solver.py)

---

**遇到问题？**
1. 检查 `runs/detect/train/` 下的训练日志
2. 查看详细文档：`docs/MODEL_USAGE.md`
3. 确认 GPU 是否可用：`python -c "import torch; print(torch.cuda.is_available())"`

**祝你拆弹顺利！💣**
