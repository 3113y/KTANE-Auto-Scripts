"""
决策系统
智能规划操作序列和模块处理顺序
"""
from typing import List, Optional, Tuple
from dataclasses import dataclass
from state_manager import (
    StateManager, BombInfo, ModuleInfo, ModuleType, 
    BombFace, AuxiliaryInfo
)


@dataclass
class Action:
    """操作指令"""
    action_type: str  # "scan", "rotate", "click", "collect_info", "solve"
    target: any       # 目标（面、模块、坐标等）
    parameters: dict = None
    
    def __str__(self):
        return f"{self.action_type}({self.target})"


class DecisionMaker:
    """决策制定器"""
    
    def __init__(self, state_manager: StateManager):
        self.state = state_manager
        
        # 模块优先级（某些模块应优先处理）
        self.module_priority = {
            ModuleType.LINE: 2,
            ModuleType.BUTTON: 2,
            ModuleType.FOUR_COLOR: 3,
            ModuleType.MEMORY: 1,  # 记忆模块较复杂，后处理
            ModuleType.MAZE: 3,
        }
        
        # 辅助信息重要性（优先收集的信息）
        self.info_priority = {
            "serial_number": 1,    # 最重要
            "battery_count": 2,
            "has_car_indicator": 3,
            "has_frk_indicator": 3,
        }
    
    def plan_next_action(self) -> Optional[Action]:
        """
        规划下一步行动
        
        Returns:
            下一步操作，如果没有则返回None（表示完成）
        """
        # 1. 如果还没扫描完所有面 -> 扫描
        if not self.state.is_scanning_complete():
            return self._plan_scan_action()
        
        # 2. 如果缺少必要信息 -> 收集信息
        missing_info = self._get_missing_critical_info()
        if missing_info:
            return self._plan_collect_info_action(missing_info)
        
        # 3. 选择下一个要解的模块
        next_module = self._select_next_module()
        if next_module:
            return self._plan_solve_module_action(next_module)
        
        # 4. 所有模块都完成了
        return None
    
    def _plan_scan_action(self) -> Action:
        """规划扫描行动"""
        # 找到未扫描的面
        all_faces = set(BombFace)
        scanned = self.state.bomb_info.scanned_faces
        unscanned = all_faces - scanned
        
        if unscanned:
            # 选择下一个要扫描的面（优先前后面）
            priority_order = [
                BombFace.FRONT, BombFace.BACK,
                BombFace.TOP, BombFace.BOTTOM,
                BombFace.LEFT, BombFace.RIGHT
            ]
            
            for face in priority_order:
                if face in unscanned:
                    return Action(
                        action_type="scan_face",
                        target=face
                    )
        
        return None
    
    def _get_missing_critical_info(self) -> List[str]:
        """获取缺失的关键信息"""
        # 找出所有模块需要的信息
        all_modules = self.state.bomb_info.get_all_modules()
        required_info = set()
        
        for module in all_modules:
            required_info.update(module.required_info)
        
        # 找出还未收集的
        collected = self.state.bomb_info.auxiliary.collected
        missing = required_info - collected
        
        # 按优先级排序
        missing_sorted = sorted(
            missing,
            key=lambda x: self.info_priority.get(x, 99)
        )
        
        return missing_sorted
    
    def _plan_collect_info_action(self, missing_info: List[str]) -> Action:
        """规划收集信息行动"""
        # 选择第一个缺失的信息
        info_type = missing_info[0]
        
        # 确定信息所在的面
        info_location_map = {
            "serial_number": [BombFace.TOP, BombFace.BOTTOM],
            "battery_count": [BombFace.LEFT, BombFace.RIGHT],
            "has_car_indicator": [BombFace.LEFT, BombFace.RIGHT],
            "has_frk_indicator": [BombFace.LEFT, BombFace.RIGHT],
        }
        
        target_faces = info_location_map.get(info_type, [])
        
        return Action(
            action_type="collect_info",
            target=info_type,
            parameters={"faces": target_faces}
        )
    
    def _select_next_module(self) -> Optional[ModuleInfo]:
        """
        选择下一个要解的模块
        使用智能排序策略
        """
        # 获取所有待处理模块
        pending = self.state.bomb_info.get_pending_modules()
        
        if not pending:
            return None
        
        # 过滤出信息已足够的模块
        solvable = [
            m for m in pending
            if self.state.bomb_info.auxiliary.is_complete(m.required_info)
        ]
        
        if not solvable:
            return None
        
        # 按优先级排序
        solvable_sorted = sorted(
            solvable,
            key=lambda m: self.module_priority.get(m.module_type, 99)
        )
        
        return solvable_sorted[0]
    
    def _plan_solve_module_action(self, module: ModuleInfo) -> Action:
        """规划解模块行动"""
        return Action(
            action_type="solve_module",
            target=module
        )
    
    def generate_action_sequence(self) -> List[Action]:
        """
        生成完整的操作序列
        
        Returns:
            操作列表
        """
        actions = []
        
        # 模拟规划多步
        max_steps = 50
        for _ in range(max_steps):
            action = self.plan_next_action()
            if action is None:
                break
            actions.append(action)
            
            # 模拟执行（更新状态）
            # 实际使用时由执行器负责
            
        return actions
    
    def explain_strategy(self):
        """解释当前策略"""
        print("\n策略分析:")
        print("-" * 40)
        
        # 扫描进度
        scanned = len(self.state.bomb_info.scanned_faces)
        print(f"1. 扫描进度: {scanned}/6 面")
        
        # 信息收集
        missing = self._get_missing_critical_info()
        if missing:
            print(f"2. 缺失信息: {', '.join(missing)}")
        else:
            print("2. 信息收集: 完整")
        
        # 模块队列
        pending = self.state.bomb_info.get_pending_modules()
        print(f"3. 待处理模块: {len(pending)}")
        
        if pending:
            for i, module in enumerate(pending[:3], 1):
                priority = self.module_priority.get(module.module_type, 99)
                print(f"   {i}. {module.module_type.value} (优先级: {priority})")
        
        print("-" * 40)


def test_decision_maker():
    """测试决策系统"""
    from state_manager import StateManager, ModuleInfo, ModuleType, BombFace
    
    print("测试决策系统...\n")
    
    # 创建测试环境
    state = StateManager()
    
    # 添加模块
    modules = [
        (BombFace.FRONT, ModuleType.LINE, {"serial_number"}),
        (BombFace.FRONT, ModuleType.BUTTON, {"battery_count"}),
        (BombFace.BACK, ModuleType.MAZE, set()),
    ]
    
    for face, mtype, required in modules:
        module = ModuleInfo(module_type=mtype, required_info=required)
        state.bomb_info.add_module(face, module)
    
    # 创建决策器
    decision = DecisionMaker(state)
    
    # 测试规划
    print("1. 初始状态分析:")
    decision.explain_strategy()
    
    print("\n2. 规划操作序列:")
    for i in range(5):
        action = decision.plan_next_action()
        if action:
            print(f"   步骤 {i+1}: {action}")
            
            # 模拟执行
            if action.action_type == "scan_face":
                state.mark_face_scanned(action.target)
            elif action.action_type == "collect_info":
                state.bomb_info.auxiliary.collected.add(action.target)
        else:
            print("   完成！")
            break
    
    print("\n测试完成！")


if __name__ == "__main__":
    test_decision_maker()
