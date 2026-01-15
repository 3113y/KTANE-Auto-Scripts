"""
状态管理系统
跟踪炸弹信息、模块状态、视图状态
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
from enum import Enum


class ViewState(Enum):
    """视图状态"""
    MENU = "menu"
    MAIN_VIEW = "main_view"        # 主视图（可旋转盒子）
    MODULE_VIEW = "module_view"    # 模块放大视图
    UNKNOWN = "unknown"


class BombFace(Enum):
    """炸弹盒子的面"""
    FRONT = "front"
    BACK = "back"
    TOP = "top"
    BOTTOM = "bottom"
    LEFT = "left"
    RIGHT = "right"


class ModuleType(Enum):
    """模块类型"""
    LINE = "line_module"
    BUTTON = "button_module"
    FOUR_COLOR = "four_color_block"
    MEMORY = "memory_module"
    MAZE = "maze_module"
    UNKNOWN = "unknown"


class ModuleStatus(Enum):
    """模块状态"""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ModuleInfo:
    """模块信息"""
    module_type: ModuleType
    status: ModuleStatus = ModuleStatus.NOT_STARTED
    location_face: Optional[BombFace] = None
    position: Optional[tuple] = None  # (x, y) 在该面的位置
    click_position: Optional[tuple] = None  # 点击坐标（屏幕坐标）
    
    # 模块特定信息
    details: Dict = field(default_factory=dict)
    
    # 解题需要的辅助信息
    required_info: Set[str] = field(default_factory=set)


@dataclass
class AuxiliaryInfo:
    """辅助信息（序列号、电池等）"""
    serial_number: Optional[str] = None
    battery_count: int = 0
    has_car_indicator: bool = False
    has_frk_indicator: bool = False
    parallel_port_count: int = 0
    
    # 记录哪些信息已收集
    collected: Set[str] = field(default_factory=set)
    
    def is_complete(self, required: Set[str]) -> bool:
        """检查是否已收集所有必需信息"""
        return required.issubset(self.collected)
    
    def get_serial_has_vowel(self) -> Optional[bool]:
        """序列号是否包含元音字母"""
        if self.serial_number is None:
            return None
        vowels = set('AEIOU')
        return any(c in vowels for c in self.serial_number.upper())
    
    def get_serial_last_digit(self) -> Optional[int]:
        """序列号最后一位数字"""
        if self.serial_number is None:
            return None
        for c in reversed(self.serial_number):
            if c.isdigit():
                return int(c)
        return None


@dataclass
class BombInfo:
    """完整炸弹信息"""
    # 各面的模块
    face_modules: Dict[BombFace, List[ModuleInfo]] = field(default_factory=dict)
    
    # 辅助信息
    auxiliary: AuxiliaryInfo = field(default_factory=AuxiliaryInfo)
    
    # 已扫描的面
    scanned_faces: Set[BombFace] = field(default_factory=set)
    
    def add_module(self, face: BombFace, module: ModuleInfo):
        """添加模块"""
        if face not in self.face_modules:
            self.face_modules[face] = []
        self.face_modules[face].append(module)
        module.location_face = face
    
    def get_all_modules(self) -> List[ModuleInfo]:
        """获取所有模块"""
        modules = []
        for face_modules in self.face_modules.values():
            modules.extend(face_modules)
        return modules
    
    def get_pending_modules(self) -> List[ModuleInfo]:
        """获取待处理的模块"""
        return [m for m in self.get_all_modules() 
                if m.status == ModuleStatus.NOT_STARTED]
    
    def get_next_solvable_module(self) -> Optional[ModuleInfo]:
        """
        获取下一个可解的模块
        （已收集足够信息的模块）
        """
        pending = self.get_pending_modules()
        
        for module in pending:
            if self.auxiliary.is_complete(module.required_info):
                return module
        
        return None
    
    def is_all_info_collected(self) -> bool:
        """是否已收集所有必需信息"""
        all_modules = self.get_all_modules()
        all_required = set()
        
        for module in all_modules:
            all_required.update(module.required_info)
        
        return self.auxiliary.is_complete(all_required)
    
    def is_all_completed(self) -> bool:
        """是否所有模块都已完成"""
        all_modules = self.get_all_modules()
        return all(m.status == ModuleStatus.COMPLETED for m in all_modules)


@dataclass
class StateManager:
    """状态管理器"""
    
    # 当前视图状态
    current_view: ViewState = ViewState.UNKNOWN
    current_face: BombFace = BombFace.FRONT
    
    # 炸弹信息
    bomb_info: BombInfo = field(default_factory=BombInfo)
    
    # 当前正在处理的模块
    current_module: Optional[ModuleInfo] = None
    
    # 历史记录
    operation_history: List[str] = field(default_factory=list)
    
    def log_operation(self, operation: str):
        """记录操作"""
        self.operation_history.append(operation)
        print(f"[日志] {operation}")
    
    def update_view_state(self, new_state: ViewState):
        """更新视图状态"""
        if new_state != self.current_view:
            self.log_operation(f"视图变化: {self.current_view.value} -> {new_state.value}")
            self.current_view = new_state
    
    def update_current_face(self, new_face: BombFace):
        """更新当前面"""
        if new_face != self.current_face:
            self.log_operation(f"旋转到: {new_face.value}")
            self.current_face = new_face
    
    def mark_face_scanned(self, face: BombFace):
        """标记面已扫描"""
        self.bomb_info.scanned_faces.add(face)
        self.log_operation(f"已扫描: {face.value}")
    
    def is_scanning_complete(self) -> bool:
        """是否完成所有面的扫描"""
        # 至少扫描前后两面（模块所在面）
        required_faces = {BombFace.FRONT, BombFace.BACK}
        return required_faces.issubset(self.bomb_info.scanned_faces)
    
    def get_progress(self) -> Dict:
        """获取进度信息"""
        all_modules = self.bomb_info.get_all_modules()
        completed = sum(1 for m in all_modules if m.status == ModuleStatus.COMPLETED)
        
        return {
            "total_modules": len(all_modules),
            "completed_modules": completed,
            "pending_modules": len(all_modules) - completed,
            "scanned_faces": len(self.bomb_info.scanned_faces),
            "info_collected": len(self.bomb_info.auxiliary.collected),
        }
    
    def print_status(self):
        """打印当前状态"""
        print("\n" + "="*50)
        print("当前状态")
        print("="*50)
        print(f"视图: {self.current_view.value}")
        print(f"当前面: {self.current_face.value}")
        
        progress = self.get_progress()
        print(f"\n进度:")
        print(f"  模块: {progress['completed_modules']}/{progress['total_modules']}")
        print(f"  已扫描面: {progress['scanned_faces']}/6")
        print(f"  已收集信息: {progress['info_collected']}")
        
        if self.current_module:
            print(f"\n当前模块: {self.current_module.module_type.value}")
            print(f"  状态: {self.current_module.status.value}")
        
        print("="*50 + "\n")


def test_state_manager():
    """测试状态管理器"""
    print("测试状态管理器...\n")
    
    # 创建状态管理器
    state = StateManager()
    
    # 添加一些测试数据
    print("1. 添加模块...")
    module1 = ModuleInfo(
        module_type=ModuleType.LINE,
        required_info={"serial_number"}
    )
    state.bomb_info.add_module(BombFace.FRONT, module1)
    
    module2 = ModuleInfo(
        module_type=ModuleType.BUTTON,
        required_info={"battery_count", "has_car_indicator"}
    )
    state.bomb_info.add_module(BombFace.FRONT, module2)
    
    # 打印状态
    state.print_status()
    
    # 收集信息
    print("\n2. 收集辅助信息...")
    state.bomb_info.auxiliary.serial_number = "ABC123"
    state.bomb_info.auxiliary.collected.add("serial_number")
    state.bomb_info.auxiliary.battery_count = 2
    state.bomb_info.auxiliary.collected.add("battery_count")
    
    # 查找可解模块
    print("\n3. 查找可解模块...")
    next_module = state.bomb_info.get_next_solvable_module()
    if next_module:
        print(f"可解: {next_module.module_type.value}")
        state.current_module = next_module
    
    # 完成模块
    print("\n4. 完成模块...")
    module1.status = ModuleStatus.COMPLETED
    
    # 最终状态
    state.print_status()
    
    print("\n测试完成！")


if __name__ == "__main__":
    test_state_manager()
