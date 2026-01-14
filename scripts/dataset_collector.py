"""
数据集采集脚本 - 用于自动截取游戏界面
使用方法：运行脚本后按F8截图，按ESC退出
"""
import cv2
import numpy as np
import pyautogui
from datetime import datetime
import os
from pathlib import Path
import keyboard
import time

class DatasetCollector:
    def __init__(self, output_dir="dataset/raw_images", max_image_size=960):
        """初始化数据集采集器
        
        Args:
            output_dir: 输出目录
            max_image_size: 图片最大边长，超过会自动缩放（默认960，YOLOv11推荐尺寸）
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.screenshot_count = 0
        self.is_running = True
        self.max_image_size = max_image_size
        
        print(f"数据集保存路径: {self.output_dir.absolute()}")
        print(f"图片最大尺寸: {max_image_size}px (自动缩放)")
        print("=" * 50)
        print("控制说明:")
        print("  F8  - 截取屏幕")
        print("  F9  - 截取指定区域（拖动选择）")
        print("  ESC - 退出程序")
        print("=" * 50)
    
    def capture_fullscreen(self):
        """截取全屏"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"screen_{timestamp}_{self.screenshot_count:04d}.png"
        filepath = self.output_dir / filename
        
        # 使用pyautogui截图
        screenshot = pyautogui.screenshot()
        screenshot = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
        
        # 自动缩放过大图片
        screenshot = self._resize_if_needed(screenshot)
        
        cv2.imwrite(str(filepath), screenshot)
        self.screenshot_count += 1
        print(f"✓ 已保存: {filename} ({screenshot.shape[1]}x{screenshot.shape[0]})")
        
        # 显示缩略图（可选）
        self.show_thumbnail(screenshot, filename)
    
    def capture_region(self):
        """交互式选择区域截图"""
        print("\n请在游戏窗口上拖动鼠标选择区域...")
        time.sleep(0.5)
        
        # 全屏截图作为底图
        screenshot = pyautogui.screenshot()
        screenshot = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
        
        # 创建选择窗口
        clone = screenshot.copy()
        cv2.namedWindow("选择区域 - 拖动鼠标选择，按ENTER确认，按ESC取消", cv2.WINDOW_NORMAL)
        
        # 设置鼠标回调
        roi = cv2.selectROI("选择区域 - 拖动鼠标选择，按ENTER确认，按ESC取消", screenshot, False)
        cv2.destroyAllWindows()
        
        if roi[2] > 0 and roi[3] > 0:  # 确保选择了有效区域
            x, y, w, h = roi
            cropped = screenshot[y:y+h, x:x+w]
            
            # 自动缩放过大图片
            cropped = self._resize_if_needed(cropped)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"region_{timestamp}_{self.screenshot_count:04d}.png"
            filepath = self.output_dir / filename
            
            cv2.imwrite(str(filepath), cropped)
            self.screenshot_count += 1
            print(f"✓ 已保存区域: {filename} ({cropped.shape[1]}x{cropped.shape[0]})")
        else:
            print("✗ 取消截图")
    
    def show_thumbnail(self, image, filename, max_width=400):
        """显示缩略图预览（可选）"""
        h, w = image.shape[:2]
        if w > max_width:
            scale = max_width / w
            new_w = int(w * scale)
            new_h = int(h * scale)
            thumbnail = cv2.resize(image, (new_w, new_h))
        else:
            thumbnail = image
        
        cv2.imshow("最近截图预览 (按任意键关闭)", thumbnail)
        cv2.waitKey(1000)  # 显示1秒
        cv2.destroyAllWindows()
    
    def _resize_if_needed(self, image):
        """如果图片过大则缩放"""
        h, w = image.shape[:2]
        max_dim = max(h, w)
        
        if max_dim > self.max_image_size:
            scale = self.max_image_size / max_dim
            new_w = int(w * scale)
            new_h = int(h * scale)
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            print(f"  ↓ 已缩放: {w}x{h} -> {new_w}x{new_h}")
            return resized
        
        return image
    
    def run(self):
        """运行采集器"""
        print("\n准备就绪！开始采集数据...")
        
        # 注册热键
        keyboard.add_hotkey('f8', self.capture_fullscreen)
        keyboard.add_hotkey('f9', self.capture_region)
        keyboard.add_hotkey('esc', lambda: self.stop())
        
        # 主循环
        try:
            while self.is_running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass
        finally:
            print(f"\n采集完成！共采集 {self.screenshot_count} 张图片")
            print(f"保存位置: {self.output_dir.absolute()}")
    
    def stop(self):
        """停止采集"""
        self.is_running = False


if __name__ == "__main__":
    # 检查依赖
    try:
        import pyautogui
        import keyboard
    except ImportError as e:
        print(f"缺少依赖库: {e}")
        print("请运行: pip install pyautogui keyboard opencv-python")
        exit(1)
    
    collector = DatasetCollector()
    collector.run()
