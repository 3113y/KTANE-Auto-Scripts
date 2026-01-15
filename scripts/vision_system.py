"""
视觉感知系统
负责屏幕截图、目标检测、OCR识别
"""
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import mss
import mss.tools
from ultralytics import YOLO
import win32gui
import win32ui
import win32con
from PIL import Image

@dataclass
class DetectionResult:
    """检测结果"""
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    center: Tuple[int, int]


class VisionSystem:
    """视觉感知系统"""
    
    def __init__(self, 
                 model_path: str = "runs/detect/train/weights/best.pt",
                 target_size: int = 960,
                 window_title: Optional[str] = None):
        """初始化视觉系统
        
        Args:
            model_path: YOLO模型路径
            target_size: 推理前的目标图像尺寸（长边）
            window_title: 目标窗口标题（如"Keep Talking"），None则捕获整个屏幕
        """
        print("初始化视觉系统...")
        
        self.model_path = Path(model_path)
        self.model = None
        self.screen_capturer = mss.mss()
        self.monitor = None  # 自动检测主显示器
        self.target_size = target_size
        self.window_title = window_title
        self.window_hwnd = None  # 窗口句柄
        
        # OCR引擎（可选：tesseract、easyocr等）
        self.ocr_engine = None
        
        # 如果指定了窗口标题，查找窗口
        if window_title:
            self._find_window()
        
        # 加载YOLO模型
        self._load_model()
        
    def _find_window(self):
        """查找目标窗口"""
        def callback(hwnd, windows):
            try:
                if win32gui.IsWindowVisible(hwnd):
                    title = win32gui.GetWindowText(hwnd)
                    if title and self.window_title.lower() in title.lower():
                        windows.append((hwnd, title))
            except:
                pass  # 忽略无效窗口
            return True  # 始终返回True继续枚举
        
        windows = []
        win32gui.EnumWindows(callback, windows)
        
        if windows:
            self.window_hwnd = windows[0][0]
            print(f"找到窗口: {windows[0][1]} (句柄: {self.window_hwnd})")
        else:
            print(f"警告：未找到包含 '{self.window_title}' 的窗口")
            print("将使用全屏捕获")
    
    def _load_model(self):
        """加载YOLO模型"""
        if self.model_path.exists():
            print(f"加载模型: {self.model_path}")
            self.model = YOLO(str(self.model_path))
            print(f"图像预处理: 压缩至 {self.target_size}px")
        else:
            print(f"警告：模型文件不存在 {self.model_path}")
            print("请先训练模型或指定正确的模型路径")
    
    def capture_screen(self, region: Optional[Dict] = None) -> np.ndarray:
        """
        截取屏幕或窗口
        
        Args:
            region: 截取区域 {"top": y, "left": x, "width": w, "height": h}
                   如果为None，根据配置截取窗口或屏幕
        
        Returns:
            BGR格式的图像数组
        """
        # 如果指定了窗口，使用窗口捕获
        if self.window_hwnd and region is None:
            return self._capture_window()
        
        # 否则使用屏幕捕获
        if region is None:
            # 使用主显示器（索引1）
            region = self.screen_capturer.monitors[1]
        
        # 截图
        screenshot = self.screen_capturer.grab(region)
        
        # 转换为numpy数组（BGRA -> BGR）
        img = np.array(screenshot)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        
        return img
    
    def _capture_window(self) -> np.ndarray:
        """捕获指定窗口"""
        try:
            # 获取窗口区域
            left, top, right, bottom = win32gui.GetWindowRect(self.window_hwnd)
            width = right - left
            height = bottom - top
            
            # 获取窗口DC
            hwnd_dc = win32gui.GetWindowDC(self.window_hwnd)
            mfc_dc = win32ui.CreateDCFromHandle(hwnd_dc)
            save_dc = mfc_dc.CreateCompatibleDC()
            
            # 创建位图
            save_bitmap = win32ui.CreateBitmap()
            save_bitmap.CreateCompatibleBitmap(mfc_dc, width, height)
            save_dc.SelectObject(save_bitmap)
            
            # 复制窗口内容
            save_dc.BitBlt((0, 0), (width, height), mfc_dc, (0, 0), win32con.SRCCOPY)
            
            # 转换为numpy数组
            bmpinfo = save_bitmap.GetInfo()
            bmpstr = save_bitmap.GetBitmapBits(True)
            img = np.frombuffer(bmpstr, dtype=np.uint8)
            img = img.reshape((height, width, 4))
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            
            # 清理
            win32gui.DeleteObject(save_bitmap.GetHandle())
            save_dc.DeleteDC()
            mfc_dc.DeleteDC()
            win32gui.ReleaseDC(self.window_hwnd, hwnd_dc)
            
            return img
            
        except Exception as e:
            print(f"窗口捕获失败: {e}，切换到全屏捕获")
            self.window_hwnd = None  # 禁用窗口捕获
            return self.capture_screen()
    
    def _preprocess_image(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """预处理图像：压缩到目标尺寸
        
        Args:
            image: 原始图像
            
        Returns:
            (处理后的图像, 缩放比例)
        """
        h, w = image.shape[:2]
        
        # 计算缩放比例
        if max(h, w) > self.target_size:
            scale = self.target_size / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            return resized, scale
        
        return image, 1.0
    
    def detect_objects(self, image: np.ndarray, 
                      conf_threshold: float = 0.5,
                      preprocess: bool = True) -> List[DetectionResult]:
        """
        使用YOLO检测目标
        
        Args:
            image: 输入图像
            conf_threshold: 置信度阈值
            preprocess: 是否预处理图像（压缩）
        
        Returns:
            检测结果列表（坐标已还原到原始图像尺寸）
        """
        if self.model is None:
            print("错误：模型未加载")
            return []
        
        # 预处理图像
        if preprocess:
            processed_img, scale = self._preprocess_image(image)
        else:
            processed_img = image
            scale = 1.0
        
        # YOLO推理
        results = self.model(processed_img, conf=conf_threshold, verbose=False)
        
        # 解析结果
        detections = []
        for result in results:
            boxes = result.boxes
            for i in range(len(boxes)):
                # 获取边界框（在缩放后的图像上）
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                
                # 还原到原始图像坐标
                if scale != 1.0:
                    x1, y1, x2, y2 = x1/scale, y1/scale, x2/scale, y2/scale
                
                # 获取类别和置信度
                class_id = int(boxes.cls[i])
                confidence = float(boxes.conf[i])
                class_name = result.names[class_id]
                
                # 计算中心点
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                
                detection = DetectionResult(
                    class_id=class_id,
                    class_name=class_name,
                    confidence=confidence,
                    bbox=(int(x1), int(y1), int(x2), int(y2)),
                    center=(center_x, center_y)
                )
                detections.append(detection)
        
        return detections
    
    def recognize_text(self, image: np.ndarray, 
                      region: Optional[Tuple[int, int, int, int]] = None) -> str:
        """
        OCR文字识别
        
        Args:
            image: 输入图像
            region: ROI区域 (x1, y1, x2, y2)
        
        Returns:
            识别的文字
        """
        # TODO: 实现OCR
        # 可以使用 pytesseract、easyocr、PaddleOCR等
        
        if region:
            x1, y1, x2, y2 = region
            roi = image[y1:y2, x1:x2]
        else:
            roi = image
        
        # 预处理：灰度化、二值化等
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # OCR识别（需要安装对应库）
        text = ""
        try:
            import pytesseract
            text = pytesseract.image_to_string(binary)
        except ImportError:
            print("提示：未安装pytesseract，OCR功能不可用")
        
        return text.strip()
    
    def classify_view_state(self, detections: List[DetectionResult]) -> str:
        """
        根据检测结果判断当前视图状态
        
        Returns:
            "main_view" - 主视图（盒子外观）
            "module_view" - 模块放大视图
            "menu" - 菜单界面
        """
        # TODO: 实现视图分类逻辑
        # 可以基于：
        # 1. 检测到的对象类型和数量
        # 2. 布局特征
        # 3. 特定UI元素
        
        # 简单规则示例
        if len(detections) == 0:
            return "menu"
        
        # 如果只检测到一个大模块，可能是放大视图
        if len(detections) == 1 and detections[0].class_name.endswith("_module"):
            bbox = detections[0].bbox
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            # 如果模块占据大部分屏幕
            if area > 200000:  # 根据实际调整
                return "module_view"
        
        return "main_view"
    
    def visualize_detections(self, image: np.ndarray, 
                           detections: List[DetectionResult],
                           window_name: str = "Detections") -> np.ndarray:
        """
        可视化检测结果
        
        Args:
            image: 原始图像
            detections: 检测结果
            window_name: 窗口名称
        
        Returns:
            标注后的图像
        """
        img_vis = image.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            
            # 绘制边界框
            color = (0, 255, 0)
            cv2.rectangle(img_vis, (x1, y1), (x2, y2), color, 2)
            
            # 绘制标签
            label = f"{det.class_name} {det.confidence:.2f}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img_vis, (x1, y1 - h - 4), (x1 + w, y1), color, -1)
            cv2.putText(img_vis, label, (x1, y1 - 2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            # 绘制中心点
            cv2.circle(img_vis, det.center, 5, (0, 0, 255), -1)
        
        return img_vis


def test_vision_system():
    """测试视觉系统"""
    print("测试视觉系统...")
    
    vision = VisionSystem()
    
    print("\n1. 截取屏幕测试...")
    img = vision.capture_screen()
    print(f"截图尺寸: {img.shape}")
    
    print("\n2. 目标检测测试...")
    detections = vision.detect_objects(img)
    print(f"检测到 {len(detections)} 个目标")
    
    for det in detections:
        print(f"  - {det.class_name}: {det.confidence:.2f} at {det.center}")
    
    print("\n3. 视图分类测试...")
    view_state = vision.classify_view_state(detections)
    print(f"当前视图: {view_state}")
    
    print("\n4. 可视化...")
    img_vis = vision.visualize_detections(img, detections)
    cv2.imshow("Vision Test", img_vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    test_vision_system()
