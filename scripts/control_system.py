"""
控制系统
负责鼠标和键盘操作
"""
import time
import pyautogui
from typing import Tuple, Optional
from dataclasses import dataclass
import win32gui
import win32con


@dataclass  
class ControlSystem:
    """游戏控制系统"""
    
    # 游戏窗口
    game_window_title: str = "Keep Talking"
    game_hwnd: Optional[int] = None
    
    # 鼠标设置
    mouse_speed: float = 0.3  # 移动持续时间
    click_delay: float = 0.1  # 点击后延迟
    
    # 旋转参数
    rotation_sensitivity: int = 200  # 旋转灵敏度（像素）
    
    def __post_init__(self):
        """初始化"""
        print("初始化控制系统...")
        
        # PyAutoGUI安全设置
        pyautogui.FAILSAFE = True  # 鼠标移到角落可紧急停止
        pyautogui.PAUSE = 0.05     # 每个操作后的暂停
        
        # 查找游戏窗口
        self._find_game_window()
    
    def _find_game_window(self):
        """查找游戏窗口"""
        def enum_windows_callback(hwnd, windows):
            if win32gui.IsWindowVisible(hwnd):
                title = win32gui.GetWindowText(hwnd)
                if self.game_window_title.lower() in title.lower():
                    windows.append((hwnd, title))
        
        windows = []
        win32gui.EnumWindows(enum_windows_callback, windows)
        
        if windows:
            self.game_hwnd = windows[0][0]
            print(f"找到游戏窗口: {windows[0][1]}")
        else:
            print(f"警告：未找到游戏窗口（标题包含'{self.game_window_title}'）")
    
    def activate_game_window(self):
        """激活游戏窗口"""
        if self.game_hwnd:
            try:
                win32gui.SetForegroundWindow(self.game_hwnd)
                time.sleep(0.2)
                return True
            except Exception as e:
                print(f"激活窗口失败: {e}")
        return False
    
    def get_game_window_rect(self) -> Optional[Tuple[int, int, int, int]]:
        """获取游戏窗口位置和大小"""
        if self.game_hwnd:
            try:
                rect = win32gui.GetWindowRect(self.game_hwnd)
                return rect  # (left, top, right, bottom)
            except Exception as e:
                print(f"获取窗口位置失败: {e}")
        return None
    
    def click_at(self, x: int, y: int, button: str = "left", clicks: int = 1):
        """
        在指定位置点击
        
        Args:
            x, y: 屏幕坐标
            button: "left" 或 "right"
            clicks: 点击次数
        """
        print(f"点击: ({x}, {y})")
        
        # 移动鼠标
        pyautogui.moveTo(x, y, duration=self.mouse_speed)
        time.sleep(0.1)
        
        # 点击
        pyautogui.click(x, y, clicks=clicks, button=button)
        time.sleep(self.click_delay)
    
    def drag_to_rotate(self, direction: str, distance: Optional[int] = None):
        """
        拖动鼠标旋转盒子
        
        Args:
            direction: "left", "right", "up", "down"
            distance: 拖动距离（像素），None则使用默认
        """
        if distance is None:
            distance = self.rotation_sensitivity
        
        # 获取当前鼠标位置
        current_x, current_y = pyautogui.position()
        
        # 计算目标位置
        if direction == "left":
            target_x = current_x - distance
            target_y = current_y
        elif direction == "right":
            target_x = current_x + distance
            target_y = current_y
        elif direction == "up":
            target_x = current_x
            target_y = current_y - distance
        elif direction == "down":
            target_x = current_x
            target_y = current_y + distance
        else:
            print(f"错误：未知方向 {direction}")
            return
        
        print(f"旋转盒子: {direction} ({distance}px)")
        
        # 拖动
        pyautogui.drag(target_x - current_x, target_y - current_y, 
                      duration=0.5, button='left')
        time.sleep(0.3)
    
    def rotate_to_face(self, face: str):
        """
        旋转到指定面
        
        Args:
            face: "front", "back", "top", "bottom", "left", "right"
        """
        print(f"旋转到: {face}")
        
        # TODO: 实现智能旋转
        # 需要记录当前面方向，计算最短旋转路径
        
        # 简化版本：固定旋转序列
        rotation_map = {
            "back": [("right", 2)],      # 旋转180度
            "top": [("up", 1)],           # 向上翻
            "bottom": [("down", 1)],      # 向下翻  
            "left": [("left", 1)],        # 左转90度
            "right": [("right", 1)],      # 右转90度
        }
        
        if face == "front":
            print("已在正面")
            return
        
        if face in rotation_map:
            for direction, count in rotation_map[face]:
                for _ in range(count):
                    self.drag_to_rotate(direction)
                    time.sleep(0.5)
    
    def press_key(self, key: str, hold_time: float = 0.1):
        """
        按键
        
        Args:
            key: 键名（如 "space", "esc", "a"）
            hold_time: 按住时间
        """
        print(f"按键: {key}")
        pyautogui.keyDown(key)
        time.sleep(hold_time)
        pyautogui.keyUp(key)
        time.sleep(0.1)
    
    def type_text(self, text: str, interval: float = 0.1):
        """
        输入文字
        
        Args:
            text: 要输入的文字
            interval: 字符间隔
        """
        print(f"输入文字: {text}")
        pyautogui.write(text, interval=interval)
        time.sleep(0.2)
    
    def move_mouse_smooth(self, x: int, y: int, duration: float = 0.5):
        """平滑移动鼠标"""
        pyautogui.moveTo(x, y, duration=duration)
    
    def get_mouse_position(self) -> Tuple[int, int]:
        """获取当前鼠标位置"""
        return pyautogui.position()
    
    def wait(self, seconds: float):
        """等待"""
        time.sleep(seconds)


def test_control_system():
    """测试控制系统"""
    print("测试控制系统...")
    print("请确保游戏窗口已打开")
    input("按回车继续...")
    
    control = ControlSystem()
    
    # 激活窗口
    print("\n1. 激活游戏窗口...")
    control.activate_game_window()
    
    # 获取窗口位置
    print("\n2. 获取窗口位置...")
    rect = control.get_game_window_rect()
    if rect:
        print(f"窗口位置: {rect}")
        center_x = (rect[0] + rect[2]) // 2
        center_y = (rect[1] + rect[3]) // 2
        print(f"窗口中心: ({center_x}, {center_y})")
        
        # 移动鼠标到中心
        print("\n3. 移动鼠标到中心...")
        control.move_mouse_smooth(center_x, center_y)
        
        # 测试旋转
        print("\n4. 测试旋转（按Ctrl+C可中断）...")
        time.sleep(1)
        
        print("向右旋转...")
        control.drag_to_rotate("right")
        time.sleep(1)
        
        print("向左旋转...")
        control.drag_to_rotate("left")
        time.sleep(1)
        
        print("向上旋转...")
        control.drag_to_rotate("up")
        time.sleep(1)
        
        print("向下旋转...")
        control.drag_to_rotate("down")
    
    print("\n测试完成！")


if __name__ == "__main__":
    test_control_system()
