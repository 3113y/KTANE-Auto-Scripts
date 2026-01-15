"""
快速测试训练好的YOLO模型
支持：窗口捕获、图像压缩、OBS虚拟摄像头
"""
import cv2
from ultralytics import YOLO
from pathlib import Path
import mss
import numpy as np
import sys
import win32gui


def list_windows():
    """列出所有可见窗口（排除最小化窗口）"""
    windows = []
    
    def callback(hwnd, _):
        try:
            if win32gui.IsWindowVisible(hwnd):
                # 排除最小化的窗口
                if win32gui.IsIconic(hwnd):
                    return True
                
                title = win32gui.GetWindowText(hwnd)
                if title and len(title) > 0:  # 只显示有标题的窗口
                    # 获取窗口大小，过滤掉太小的窗口
                    rect = win32gui.GetWindowRect(hwnd)
                    width = rect[2] - rect[0]
                    height = rect[3] - rect[1]
                    if width > 100 and height > 100:  # 过滤小窗口
                        windows.append((hwnd, title, width, height))
        except:
            pass
        return True
    
    win32gui.EnumWindows(callback, None)
    return windows


def select_capture_source():
    """
    选择捕获源
    
    Returns:
        ("screen", None) - 全屏捕获
        ("window", "窗口标题") - 窗口捕获
        ("camera", camera_id) - OBS虚拟摄像头
    """
    print("\n" + "=" * 60)
    print("选择捕获源")
    print("=" * 60)
    print("1. 全屏捕获（主显示器）")
    print("2. 窗口捕获（只捕获游戏窗口）")
    print("3. OBS虚拟摄像头")
    
    choice = input("\n请选择 [1-3，默认2]: ").strip() or "2"
    
    if choice == "1":
        print("✓ 使用全屏捕获")
        return ("screen", None)
    
    elif choice == "2":
        print("\n正在列出可见窗口...")
        windows = list_windows()
        
        # 按窗口大小排序（大窗口优先）
        windows.sort(key=lambda x: x[2] * x[3], reverse=True)
        
        # 过滤出可能是游戏窗口的
        game_windows = [(h, t, w, ht) for h, t, w, ht in windows 
                       if any(kw in t.lower() for kw in 
                              ["keep talking", "ktane", "bomb", "steam", "unity", "game"])]
        
        if game_windows:
            print("\n找到可能的游戏窗口：")
            for i, (hwnd, title, w, h) in enumerate(game_windows, 1):
                print(f"{i}. {title} ({w}x{h})")
            
            choice = input(f"\n选择窗口 [1-{len(game_windows)}，默认1]: ").strip() or "1"
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(game_windows):
                    window_title = game_windows[idx][1]
                    print(f"✓ 使用窗口: {window_title}")
                    return ("window", window_title)
            except ValueError:
                pass
        
        # 显示所有窗口供手动选择
        print("\n所有可见窗口（按大小排序）：")
        for i, (hwnd, title, w, h) in enumerate(windows[:20], 1):  # 只显示前20个
            print(f"{i}. {title} ({w}x{h})")
        
        choice = input(f"\n选择窗口 [1-{min(20, len(windows))}]: ").strip()
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(windows):
                window_title = windows[idx][1]
                print(f"✓ 使用窗口: {window_title}")
                return ("window", window_title)
        except ValueError:
            pass
        
        print("⚠ 未选择有效窗口，使用全屏捕获")
        return ("screen", None)
    
    elif choice == "3":
        print("\n检测OBS虚拟摄像头...")
        print("提示：请确保OBS已启动并开启虚拟摄像头")
        
        # 尝试检测摄像头
        for i in range(5):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    print(f"  摄像头 {i}: 可用 (分辨率: {frame.shape[1]}x{frame.shape[0]})")
                cap.release()
        
        camera_id = input("\n请输入摄像头编号 [默认0]: ").strip() or "0"
        try:
            camera_id = int(camera_id)
            print(f"✓ 使用摄像头 {camera_id}")
            return ("camera", camera_id)
        except ValueError:
            print("⚠ 无效输入，使用全屏捕获")
            return ("screen", None)
    
    else:
        print("⚠ 无效选项，使用全屏捕获")
        return ("screen", None)


def test_on_image(model_path: str, image_path: str = None):
    """
    在单张图片上测试模型
    
    Args:
        model_path: 训练好的模型权重路径，例如：
                    "runs/detect/train/weights/best.pt"
                    "runs/detect/train2/weights/best.pt"
        image_path: 测试图片路径（可选）
    """
    print(f"加载模型: {model_path}")
    model = YOLO(model_path)
    
    if image_path:
        # 在指定图片上测试
        print(f"测试图片: {image_path}")
        results = model(image_path, conf=0.5)
        
        # 显示结果
        for result in results:
            result.show()  # 弹出窗口显示
            result.save("test_result.jpg")  # 保存结果
            print(f"检测到 {len(result.boxes)} 个目标")
            
            # 打印每个检测结果
            for box in result.boxes:
                class_id = int(box.cls[0])
                class_name = result.names[class_id]
                confidence = float(box.conf[0])
                print(f"  - {class_name}: {confidence:.2%}")
    else:
        print("未指定图片，将实时截屏测试...")
        test_on_screen(model)


def test_on_screen(model: YOLO, 
                   conf_threshold: float = 0.5,
                   source_type: str = "screen",
                   source_param = None,
                   target_size: int = 960):
    """
    实时屏幕/窗口/摄像头检测（按ESC退出）
    
    Args:
        model: YOLO模型
        conf_threshold: 置信度阈值
        source_type: "screen", "window", "camera"
        source_param: 窗口标题或摄像头ID
        target_size: 推理前的目标尺寸
    """
    print("\n实时检测模式")
    print(f"- 捕获源: {source_type}")
    if source_param:
        print(f"- 参数: {source_param}")
    print(f"- 图像压缩: {target_size}px")
    print(f"- 置信度阈值: {conf_threshold}")
    print("- 按 ESC 退出")
    print("-" * 50)
    
    # 初始化捕获源
    if source_type == "camera":
        cap = cv2.VideoCapture(source_param)
        if not cap.isOpened():
            print(f"错误：无法打开摄像头 {source_param}")
            return
    elif source_type == "window":
        # 查找窗口
        def find_window(title_keyword):
            found_windows = []
            def callback(h, _):
                try:
                    if win32gui.IsWindowVisible(h):
                        title = win32gui.GetWindowText(h)
                        if title and title_keyword.lower() in title.lower():
                            found_windows.append(h)
                except:
                    pass  # 忽略无效窗口
                return True  # 始终返回True继续枚举
            
            win32gui.EnumWindows(callback, None)
            return found_windows[0] if found_windows else None
        
        hwnd = find_window(source_param)
        if not hwnd:
            print(f"错误：未找到窗口 '{source_param}'")
            print("切换到全屏捕获...")
            source_type = "screen"
    
    if source_type == "screen":
        sct = mss.mss()
        monitor = sct.monitors[1]  # 主显示器
    
    frame_count = 0
    fps_time = cv2.getTickCount()
    
    while True:
        # 捕获图像
        if source_type == "camera":
            ret, img = cap.read()
            if not ret:
                print("摄像头读取失败")
                break
        elif source_type == "window":
            try:
                import win32ui
                import win32con
                # 获取窗口区域
                left, top, right, bottom = win32gui.GetWindowRect(hwnd)
                width = right - left
                height = bottom - top
                
                # 截取窗口
                hwnd_dc = win32gui.GetWindowDC(hwnd)
                mfc_dc = win32ui.CreateDCFromHandle(hwnd_dc)
                save_dc = mfc_dc.CreateCompatibleDC()
                bitmap = win32ui.CreateBitmap()
                bitmap.CreateCompatibleBitmap(mfc_dc, width, height)
                save_dc.SelectObject(bitmap)
                save_dc.BitBlt((0, 0), (width, height), mfc_dc, (0, 0), win32con.SRCCOPY)
                
                # 转换为numpy
                bmpstr = bitmap.GetBitmapBits(True)
                img = np.frombuffer(bmpstr, dtype=np.uint8)
                img = img.reshape((height, width, 4))
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                
                # 清理
                win32gui.DeleteObject(bitmap.GetHandle())
                save_dc.DeleteDC()
                mfc_dc.DeleteDC()
                win32gui.ReleaseDC(hwnd, hwnd_dc)
            except Exception as e:
                print(f"窗口捕获失败: {e}")
                break
        else:  # screen
            screenshot = sct.grab(monitor)
            img = np.array(screenshot)
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        
        # 图像压缩（提升推理速度）
        h, w = img.shape[:2]
        if max(h, w) > target_size:
            scale = target_size / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            img_resized = cv2.resize(img, (new_w, new_h))
        else:
            img_resized = img
            scale = 1.0
        
        # YOLO检测（在压缩后的图像上）
        results = model(img_resized, conf=conf_threshold, verbose=False)
        
        # 绘制结果（在原始尺寸图像上）
        for result in results:
            # 将检测框缩放回原始尺寸
            boxes = result.boxes
            for box in boxes:
                # 获取坐标（在压缩图像上）
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                # 还原到原始尺寸
                if scale != 1.0:
                    x1, y1, x2, y2 = x1/scale, y1/scale, x2/scale, y2/scale
                
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # 类别和置信度
                class_id = int(box.cls[0])
                class_name = result.names[class_id]
                confidence = float(box.conf[0])
                
                # 绘制边界框
                color = (0, 255, 0)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                
                # 绘制标签
                label = f"{class_name} {confidence:.0%}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw, y1), color, -1)
                cv2.putText(img, label, (x1, y1 - 4), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            # 统计信息
            num_detections = len(boxes)
            detections_text = []
            for box in boxes:
                class_name = result.names[int(box.cls[0])]
                confidence = float(box.conf[0])
                detections_text.append(f"{class_name}({confidence:.0%})")
            
            # 在图像顶部显示信息
            info_lines = [
                f"检测: {num_detections} 个目标",
                f"原始: {img.shape[1]}x{img.shape[0]}",
                f"推理: {img_resized.shape[1]}x{img_resized.shape[0]}",
            ]
            
            if detections_text:
                info_lines.append(", ".join(detections_text[:3]))
            
            y_offset = 30
            for line in info_lines:
                cv2.putText(img, line, (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                y_offset += 30
            
            # 计算FPS
            frame_count += 1
            if frame_count % 10 == 0:
                current_time = cv2.getTickCount()
                fps = 10 / ((current_time - fps_time) / cv2.getTickFrequency())
                fps_time = current_time
                cv2.putText(img, f"FPS: {fps:.1f}", (10, img.shape[0] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 显示（可能需要缩小以适应屏幕）
        display_img = img
        if max(img.shape[:2]) > 1080:
            display_scale = 1080 / max(img.shape[:2])
            display_img = cv2.resize(img, None, fx=display_scale, fy=display_scale)
        
        cv2.imshow(f"YOLO 实时检测 - {source_type.upper()} (ESC退出)", display_img)
        
        # 按ESC退出
        if cv2.waitKey(1) & 0xFF == 27:
            break
    
    if source_type == "camera":
        cap.release()
    cv2.destroyAllWindows()


def test_on_dataset(model_path: str, test_images_folder: str = "dataset/raw_images"):
    """
    批量测试数据集中的图片
    
    Args:
        model_path: 模型路径
        test_images_folder: 测试图片文件夹
    """
    model = YOLO(model_path)
    test_path = Path(test_images_folder)
    
    if not test_path.exists():
        print(f"错误：文件夹不存在 {test_path}")
        return
    
    # 获取所有图片
    image_files = list(test_path.glob("*.jpg")) + list(test_path.glob("*.png"))
    
    if not image_files:
        print(f"未找到图片文件在 {test_path}")
        return
    
    print(f"找到 {len(image_files)} 张图片")
    print("开始批量测试...\n")
    
    total_detections = 0
    for img_path in image_files:
        print(f"测试: {img_path.name}")
        results = model(str(img_path), conf=0.5, verbose=False)
        
        for result in results:
            num_det = len(result.boxes)
            total_detections += num_det
            print(f"  检测到 {num_det} 个目标")
            
            # 保存结果到 runs/detect/test/
            result.save(filename=f"runs/detect/test/{img_path.name}")
    
    print(f"\n总计检测到 {total_detections} 个目标")
    print(f"结果已保存到 runs/detect/test/")


if __name__ == "__main__":
    print("=" * 60)
    print("YOLO 模型测试工具 - 增强版")
    print("支持：窗口捕获 | 图像压缩 | OBS虚拟摄像头")
    print("=" * 60)
    
    # 选择使用哪个训练好的模型
    model_path = "runs/detect/train/weights/best.pt"
    
    if not Path(model_path).exists():
        print(f"\n❌ 错误：模型文件不存在 {model_path}")
        print("请先训练模型：python scripts/train_yolo.py")
        sys.exit(1)
    
    print("\n选择测试模式：")
    print("1. 实时检测（屏幕/窗口/摄像头）⭐ 推荐")
    print("2. 测试单张图片")
    print("3. 批量测试数据集")
    
    choice = input("\n请输入选项 [1-3，默认1]: ").strip() or "1"
    
    try:
        if choice == "1":
            # 选择捕获源
            source_type, source_param = select_capture_source()
            
            # 加载模型
            print(f"\n加载模型: {model_path}")
            model = YOLO(model_path)
            
            # 询问压缩尺寸
            print("\n设置图像压缩（提升推理速度）")
            target_size = input("目标尺寸（长边像素）[默认960]: ").strip() or "960"
            try:
                target_size = int(target_size)
            except ValueError:
                target_size = 960
            
            # 询问置信度
            conf = input("置信度阈值 [默认0.5]: ").strip() or "0.5"
            try:
                conf = float(conf)
            except ValueError:
                conf = 0.5
            
            print(f"\n✓ 配置完成，开始实时检测...")
            print(f"  - 图像将压缩到 {target_size}px")
            print(f"  - 置信度阈值: {conf}")
            
            # 实时检测
            test_on_screen(model, 
                          conf_threshold=conf,
                          source_type=source_type,
                          source_param=source_param,
                          target_size=target_size)
            
        elif choice == "2":
            # 单张图片
            image_path = input("请输入图片路径: ").strip()
            test_on_image(model_path, image_path)
            
        elif choice == "3":
            # 批量测试
            folder = input("请输入图片文件夹路径 [默认: dataset/raw_images]: ").strip()
            folder = folder or "dataset/raw_images"
            test_on_dataset(model_path, folder)
            
        else:
            print("无效选项")
            
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n测试完成！")
