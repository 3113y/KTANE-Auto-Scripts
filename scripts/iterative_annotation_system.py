"""
迭代式自动标注系统
流程：手动标注 → 训练 → 自动标注 → 人工修正 → 重新训练 → 循环
"""
import cv2
import numpy as np
from pathlib import Path
import shutil
import json
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from ultralytics import YOLO


@dataclass
class IterationConfig:
    """迭代配置"""
    iteration_number: int = 0
    manual_samples: int = 0
    auto_samples: int = 0
    training_epochs: int = 50
    confidence_threshold: float = 0.5
    review_threshold: float = 0.7  # 低于此值需要人工审核
    timestamp: str = ""
    
    def save(self, path: Path):
        """保存配置"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(asdict(self), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load(cls, path: Path):
        """加载配置"""
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                return cls(**json.load(f))
        return cls()


class IterativeAnnotationSystem:
    """迭代标注系统"""
    
    def __init__(self, base_dir: str = "dataset"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        
        # 目录结构
        self.manual_dir = self.base_dir / "manual_annotations"  # 手动标注
        self.auto_dir = self.base_dir / "auto_annotations"      # 自动标注
        self.reviewed_dir = self.base_dir / "reviewed"          # 已审核
        self.raw_images_dir = self.base_dir / "raw_images"      # 原始图片
        self.models_dir = Path("runs/detect")                    # 模型目录
        
        # 创建目录
        for dir_path in [self.manual_dir, self.auto_dir, 
                         self.reviewed_dir, self.raw_images_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # 配置文件
        self.config_file = self.base_dir / "iteration_config.json"
        self.config = IterationConfig.load(self.config_file)
        
        # YOLO配置
        self.data_yaml = self.base_dir / "yolo" / "data.yaml"
        
        # 当前模型
        self.current_model = None
        
        print(f"初始化迭代标注系统")
        print(f"当前迭代: {self.config.iteration_number}")
        print(f"手动样本: {self.config.manual_samples}")
        print(f"自动样本: {self.config.auto_samples}")
    
    def collect_raw_images(self, interval: float = 2.0, duration: int = 60):
        """
        采集原始图片
        
        Args:
            interval: 截图间隔（秒）
            duration: 持续时间（秒）
        """
        print("\n" + "="*50)
        print("阶段1: 采集原始图片")
        print("="*50)
        print(f"间隔: {interval}秒, 持续: {duration}秒")
        print("请切换到游戏窗口...")
        input("按回车开始采集...")
        
        import time
        import mss
        
        sct = mss.mss()
        monitor = sct.monitors[1]  # 主显示器
        
        start_time = time.time()
        count = 0
        
        try:
            while time.time() - start_time < duration:
                # 截图
                screenshot = sct.grab(monitor)
                img = np.array(screenshot)
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                
                # 保存
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = f"raw_{timestamp}_{count:04d}.jpg"
                filepath = self.raw_images_dir / filename
                cv2.imwrite(str(filepath), img)
                
                count += 1
                print(f"已采集: {count} 张", end='\r')
                
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("\n用户中断")
        
        print(f"\n采集完成！共 {count} 张图片")
        print(f"保存位置: {self.raw_images_dir}")
    
    def manual_annotation_phase(self):
        """
        阶段1: 手动标注
        使用labelImg或自定义工具手动标注样本
        """
        print("\n" + "="*50)
        print("阶段1: 手动标注")
        print("="*50)
        
        # 统计未标注的图片
        raw_images = list(self.raw_images_dir.glob("*.jpg")) + \
                    list(self.raw_images_dir.glob("*.png"))
        
        # 已标注的图片（在manual_dir中有对应txt）
        annotated = {p.stem for p in self.manual_dir.glob("*.txt")}
        
        unannotated = [img for img in raw_images if img.stem not in annotated]
        
        print(f"原始图片: {len(raw_images)}")
        print(f"已标注: {len(annotated)}")
        print(f"未标注: {len(unannotated)}")
        
        if unannotated:
            print(f"\n建议：")
            print(f"1. 使用 labelImg 标注工具:")
            print(f"   labelImg {self.raw_images_dir} {self.manual_dir}")
            print(f"2. 或使用我们的标注工具:")
            print(f"   python scripts/yolo_editor.py")
            print(f"\n至少标注 20-30 张图片后再进行训练")
            
            # 可选：启动标注工具
            choice = input("\n是否启动标注工具? (y/n): ")
            if choice.lower() == 'y':
                self._launch_annotation_tool()
        else:
            print("所有图片都已标注！")
        
        # 更新配置
        self.config.manual_samples = len(annotated)
        self.config.save(self.config_file)
    
    def _launch_annotation_tool(self):
        """启动标注工具"""
        import subprocess
        import sys
        
        # 尝试启动 yolo_editor
        editor_path = Path("scripts/yolo_editor.py")
        if editor_path.exists():
            subprocess.run([sys.executable, str(editor_path)])
        else:
            print("标注工具不存在，请手动使用 labelImg")
    
    def prepare_training_data(self):
        """
        准备训练数据
        将手动标注和已审核的数据合并到训练集
        """
        print("\n" + "="*50)
        print("准备训练数据")
        print("="*50)
        
        yolo_dir = self.base_dir / "yolo"
        yolo_dir.mkdir(exist_ok=True)
        
        train_img_dir = yolo_dir / "images" / "train"
        train_lbl_dir = yolo_dir / "labels" / "train"
        val_img_dir = yolo_dir / "images" / "val"
        val_lbl_dir = yolo_dir / "labels" / "val"
        
        for d in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        # 收集所有已标注的数据
        all_labels = []
        
        # 手动标注的
        all_labels.extend(self.manual_dir.glob("*.txt"))
        
        # 已审核的自动标注
        all_labels.extend(self.reviewed_dir.glob("*.txt"))
        
        if not all_labels:
            print("错误：没有找到标注数据！")
            print(f"请先在以下目录放置标注文件:")
            print(f"  - {self.manual_dir}")
            print(f"  - {self.reviewed_dir}")
            return False
        
        print(f"找到 {len(all_labels)} 个标注文件")
        
        # 划分训练集和验证集 (80/20)
        np.random.shuffle(all_labels)
        split_idx = int(len(all_labels) * 0.8)
        train_labels = all_labels[:split_idx]
        val_labels = all_labels[split_idx:]
        
        print(f"训练集: {len(train_labels)}")
        print(f"验证集: {len(val_labels)}")
        
        # 复制文件
        def copy_data(labels, img_dir, lbl_dir):
            for label_path in labels:
                # 复制标注
                shutil.copy2(label_path, lbl_dir / label_path.name)
                
                # 查找对应图片
                img_name = label_path.stem
                for ext in ['.jpg', '.png', '.jpeg']:
                    img_path = self.raw_images_dir / f"{img_name}{ext}"
                    if img_path.exists():
                        shutil.copy2(img_path, img_dir / img_path.name)
                        break
        
        copy_data(train_labels, train_img_dir, train_lbl_dir)
        copy_data(val_labels, val_img_dir, val_lbl_dir)
        
        print("数据准备完成！")
        return True
    
    def train_model(self, epochs: int = 50, imgsz: int = 640):
        """
        阶段2: 训练模型
        
        Args:
            epochs: 训练轮数
            imgsz: 图片大小
        """
        print("\n" + "="*50)
        print(f"阶段2: 训练模型 (迭代 {self.config.iteration_number + 1})")
        print("="*50)
        
        # 准备数据
        if not self.prepare_training_data():
            return None
        
        # 检查 data.yaml
        if not self.data_yaml.exists():
            print(f"错误：找不到 {self.data_yaml}")
            print("请先创建 data.yaml 文件")
            return None
        
        # 选择基础模型
        if self.config.iteration_number == 0:
            # 第一次训练，从预训练模型开始
            base_model = "yolov8n.pt"
            print(f"使用预训练模型: {base_model}")
        else:
            # 后续训练，从上一次的模型继续
            prev_model = self._get_latest_model()
            if prev_model:
                base_model = prev_model
                print(f"从上一次模型继续: {base_model}")
            else:
                base_model = "yolov8n.pt"
                print(f"找不到上一次模型，使用预训练模型: {base_model}")
        
        # 训练
        print(f"\n开始训练...")
        print(f"训练轮数: {epochs}")
        print(f"图片大小: {imgsz}")
        
        model = YOLO(base_model)
        
        results = model.train(
            data=str(self.data_yaml),
            epochs=epochs,
            imgsz=imgsz,
            batch=16,
            name=f"iteration_{self.config.iteration_number + 1}",
            patience=10,
            save=True,
            plots=True,
            verbose=True
        )
        
        # 更新配置
        self.config.iteration_number += 1
        self.config.training_epochs = epochs
        self.config.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.config.save(self.config_file)
        
        # 保存模型路径
        model_path = self._get_latest_model()
        print(f"\n训练完成！")
        print(f"模型保存在: {model_path}")
        
        return model_path
    
    def _get_latest_model(self) -> Optional[Path]:
        """获取最新训练的模型"""
        # 查找最新的训练结果
        train_dirs = sorted(self.models_dir.glob("iteration_*"), 
                          key=lambda x: x.stat().st_mtime, reverse=True)
        
        if train_dirs:
            best_model = train_dirs[0] / "weights" / "best.pt"
            if best_model.exists():
                return best_model
        
        return None
    
    def auto_annotate(self, confidence_threshold: float = 0.5):
        """
        阶段3: 自动标注
        使用训练好的模型自动标注新数据
        
        Args:
            confidence_threshold: 置信度阈值
        """
        print("\n" + "="*50)
        print("阶段3: 自动标注")
        print("="*50)
        
        # 加载最新模型
        model_path = self._get_latest_model()
        if not model_path:
            print("错误：找不到训练好的模型！")
            print("请先运行训练")
            return
        
        print(f"使用模型: {model_path}")
        model = YOLO(str(model_path))
        
        # 找出未标注的图片
        raw_images = list(self.raw_images_dir.glob("*.jpg")) + \
                    list(self.raw_images_dir.glob("*.png"))
        
        # 已处理的图片
        processed = set()
        processed.update(p.stem for p in self.manual_dir.glob("*.txt"))
        processed.update(p.stem for p in self.auto_dir.glob("*.txt"))
        processed.update(p.stem for p in self.reviewed_dir.glob("*.txt"))
        
        unannotated = [img for img in raw_images if img.stem not in processed]
        
        print(f"未标注图片: {len(unannotated)}")
        
        if not unannotated:
            print("所有图片都已处理！")
            return
        
        # 自动标注
        print(f"置信度阈值: {confidence_threshold}")
        print("开始自动标注...")
        
        annotated_count = 0
        low_confidence_count = 0
        
        for img_path in unannotated:
            # 推理
            results = model(str(img_path), conf=confidence_threshold, verbose=False)
            
            # 检查是否有检测结果
            if len(results) == 0 or len(results[0].boxes) == 0:
                continue
            
            # 获取图片尺寸
            img = cv2.imread(str(img_path))
            img_h, img_w = img.shape[:2]
            
            # 转换为YOLO格式
            annotations = []
            min_confidence = 1.0
            
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                
                # 转换为YOLO格式 (x_center, y_center, width, height) 归一化
                x_center = ((x1 + x2) / 2) / img_w
                y_center = ((y1 + y2) / 2) / img_h
                width = (x2 - x1) / img_w
                height = (y2 - y1) / img_h
                
                annotations.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
                min_confidence = min(min_confidence, conf)
            
            # 保存标注
            if annotations:
                label_path = self.auto_dir / f"{img_path.stem}.txt"
                with open(label_path, 'w') as f:
                    f.write('\n'.join(annotations))
                
                # 保存置信度信息（用于后续审核）
                meta_path = self.auto_dir / f"{img_path.stem}.meta.json"
                with open(meta_path, 'w') as f:
                    json.dump({
                        'min_confidence': min_confidence,
                        'num_objects': len(annotations),
                        'timestamp': datetime.now().isoformat()
                    }, f)
                
                annotated_count += 1
                
                if min_confidence < self.config.review_threshold:
                    low_confidence_count += 1
                
                print(f"已标注: {annotated_count}, 低置信度: {low_confidence_count}", end='\r')
        
        print(f"\n\n自动标注完成！")
        print(f"成功标注: {annotated_count}")
        print(f"需要审核: {low_confidence_count}")
        
        self.config.auto_samples = annotated_count
        self.config.save(self.config_file)
    
    def review_annotations(self):
        """
        阶段4: 人工审核
        审核自动标注的结果，修正错误
        """
        print("\n" + "="*50)
        print("阶段4: 人工审核")
        print("="*50)
        
        # 找出需要审核的标注
        auto_labels = list(self.auto_dir.glob("*.txt"))
        auto_labels = [p for p in auto_labels if not p.name.endswith('.meta.json')]
        
        print(f"待审核: {len(auto_labels)}")
        
        if not auto_labels:
            print("没有需要审核的标注！")
            return
        
        # 优先审核低置信度的
        def get_priority(label_path):
            meta_path = label_path.parent / f"{label_path.stem}.meta.json"
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
                    return -meta.get('min_confidence', 1.0)  # 负数，低置信度优先
            return 0
        
        auto_labels.sort(key=get_priority)
        
        print("\n审核选项:")
        print("1. 使用可视化工具审核")
        print("2. 使用 yolo_editor 编辑器审核")
        print("3. 批量接受所有高置信度标注")
        
        choice = input("请选择 (1-3): ")
        
        if choice == '1':
            self._visual_review(auto_labels)
        elif choice == '2':
            self._editor_review()
        elif choice == '3':
            self._batch_accept(auto_labels)
    
    def _visual_review(self, labels: List[Path]):
        """可视化审核"""
        print("\n开始可视化审核...")
        print("操作说明:")
        print("  [Space] - 接受当前标注")
        print("  [D] - 删除当前标注")
        print("  [Q] - 退出审核")
        
        for idx, label_path in enumerate(labels):
            # 加载图片
            img_name = label_path.stem
            img_path = None
            for ext in ['.jpg', '.png', '.jpeg']:
                temp_path = self.raw_images_dir / f"{img_name}{ext}"
                if temp_path.exists():
                    img_path = temp_path
                    break
            
            if not img_path:
                continue
            
            img = cv2.imread(str(img_path))
            
            # 读取标注
            with open(label_path) as f:
                annotations = f.readlines()
            
            # 绘制标注
            img_h, img_w = img.shape[:2]
            img_vis = img.copy()
            
            for ann in annotations:
                parts = ann.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    x_center = float(parts[1]) * img_w
                    y_center = float(parts[2]) * img_h
                    width = float(parts[3]) * img_w
                    height = float(parts[4]) * img_h
                    
                    x1 = int(x_center - width / 2)
                    y1 = int(y_center - height / 2)
                    x2 = int(x_center + width / 2)
                    y2 = int(y_center + height / 2)
                    
                    cv2.rectangle(img_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img_vis, f"Class {cls_id}", (x1, y1-5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # 显示置信度信息
            meta_path = label_path.parent / f"{label_path.stem}.meta.json"
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
                    conf = meta.get('min_confidence', 0)
                    cv2.putText(img_vis, f"Confidence: {conf:.2f}", (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            cv2.putText(img_vis, f"{idx+1}/{len(labels)}", (10, img_h-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # 显示
            cv2.imshow("Review Annotations", img_vis)
            key = cv2.waitKey(0) & 0xFF
            
            if key == ord(' '):  # 接受
                # 移动到reviewed目录
                shutil.move(str(label_path), self.reviewed_dir / label_path.name)
                if meta_path.exists():
                    meta_path.unlink()
                print(f"接受: {label_path.name}")
                
            elif key == ord('d'):  # 删除
                label_path.unlink()
                if meta_path.exists():
                    meta_path.unlink()
                print(f"删除: {label_path.name}")
                
            elif key == ord('q'):  # 退出
                break
        
        cv2.destroyAllWindows()
        print("审核完成！")
    
    def _editor_review(self):
        """使用编辑器审核"""
        import subprocess
        import sys
        
        editor_path = Path("scripts/yolo_editor.py")
        if editor_path.exists():
            subprocess.run([sys.executable, str(editor_path)])
        else:
            print("编辑器不存在")
    
    def _batch_accept(self, labels: List[Path]):
        """批量接受高置信度标注"""
        threshold = float(input("置信度阈值 (建议0.7): ") or "0.7")
        
        accepted = 0
        for label_path in labels:
            meta_path = label_path.parent / f"{label_path.stem}.meta.json"
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
                    if meta.get('min_confidence', 0) >= threshold:
                        shutil.move(str(label_path), self.reviewed_dir / label_path.name)
                        meta_path.unlink()
                        accepted += 1
        
        print(f"已接受 {accepted} 个高置信度标注")
    
    def run_iteration(self):
        """运行完整迭代流程"""
        print("\n" + "="*60)
        print("迭代式标注系统 - 完整流程")
        print("="*60)
        
        while True:
            print("\n当前状态:")
            print(f"  迭代次数: {self.config.iteration_number}")
            print(f"  手动样本: {self.config.manual_samples}")
            print(f"  自动样本: {self.config.auto_samples}")
            
            print("\n选择操作:")
            print("1. 采集原始图片")
            print("2. 手动标注")
            print("3. 训练模型")
            print("4. 自动标注")
            print("5. 审核标注")
            print("6. 运行完整迭代 (3→4→5)")
            print("0. 退出")
            
            choice = input("\n请选择 (0-6): ")
            
            if choice == '1':
                self.collect_raw_images()
            elif choice == '2':
                self.manual_annotation_phase()
            elif choice == '3':
                self.train_model()
            elif choice == '4':
                self.auto_annotate()
            elif choice == '5':
                self.review_annotations()
            elif choice == '6':
                # 完整迭代
                self.train_model()
                self.auto_annotate()
                self.review_annotations()
            elif choice == '0':
                break


def main():
    """主函数"""
    print("""
    ╔═══════════════════════════════════════════════════════╗
    ║          迭代式自动标注系统 v1.0                      ║
    ║                                                       ║
    ║  流程：手动标注 → 训练 → 自动标注 → 审核 → 重训练    ║
    ╚═══════════════════════════════════════════════════════╝
    """)
    
    system = IterativeAnnotationSystem()
    system.run_iteration()


if __name__ == "__main__":
    main()
