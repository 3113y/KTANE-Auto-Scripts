"""
YOLO数据集准备脚本 - 组织数据集结构
使用方法：标注完成后运行此脚本自动整理文件结构
"""
import os
import shutil
from pathlib import Path
import random
import yaml
import cv2

class YOLODatasetPreparer:
    def __init__(self, raw_images_dir="dataset/raw_images", 
                 labels_dir="dataset/labels",
                 output_dir="dataset/yolo",
                 max_image_size=960):
        """
        初始化数据集准备器
        
        Args:
            raw_images_dir: 原始图片目录
            labels_dir: LabelImg标注后的txt标注文件目录
            output_dir: YOLO格式输出目录
            max_image_size: 图片最大边长，超过会自动缩放（默认960，YOLOv11推荐尺寸）
        """
        self.raw_images_dir = Path(raw_images_dir)
        self.labels_dir = Path(labels_dir)
        self.output_dir = Path(output_dir)
        self.max_image_size = max_image_size
        
        # 训练集和验证集比例
        self.train_ratio = 0.8
        
    def prepare_directories(self):
        """创建YOLO数据集目录结构"""
        dirs = [
            self.output_dir / "images" / "train",
            self.output_dir / "images" / "val",
            self.output_dir / "labels" / "train",
            self.output_dir / "labels" / "val",
        ]
        
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
        
        print("✓ 创建目录结构完成")
    
    def get_image_label_pairs(self):
        """获取图片和标注文件对"""
        image_extensions = ['.png', '.jpg', '.jpeg']
        pairs = []
        
        # 查找所有图片
        for img_file in self.raw_images_dir.iterdir():
            if img_file.suffix.lower() in image_extensions:
                # 查找对应的标注文件
                label_file = self.labels_dir / f"{img_file.stem}.txt"
                if label_file.exists():
                    pairs.append((img_file, label_file))
                else:
                    print(f"⚠ 警告: {img_file.name} 没有对应的标注文件")
        
        print(f"✓ 找到 {len(pairs)} 对已标注的图片")
        return pairs
    
    def split_dataset(self, pairs):
        """将数据集划分为训练集和验证集"""
        random.shuffle(pairs)
        split_idx = int(len(pairs) * self.train_ratio)
        
        train_pairs = pairs[:split_idx]
        val_pairs = pairs[split_idx:]
        
        print(f"✓ 数据集划分: 训练集 {len(train_pairs)} 张, 验证集 {len(val_pairs)} 张")
        return train_pairs, val_pairs
    
    def copy_files(self, pairs, subset_name):
        """复制文件到对应目录，并自动缩放过大的图片"""
        images_dir = self.output_dir / "images" / subset_name
        labels_dir = self.output_dir / "labels" / subset_name
        
        resized_count = 0
        for img_file, label_file in pairs:
            # 读取并可能缩放图片
            img = cv2.imread(str(img_file))
            if img is None:
                print(f"⚠ 警告: 无法读取 {img_file.name}")
                continue
            
            h, w = img.shape[:2]
            max_dim = max(h, w)
            
            # 如果图片过大，等比例缩小
            if max_dim > self.max_image_size:
                scale = self.max_image_size / max_dim
                new_w = int(w * scale)
                new_h = int(h * scale)
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                resized_count += 1
                # print(f"  缩放: {img_file.name} {w}x{h} -> {new_w}x{new_h}")
            
            # 保存图片
            cv2.imwrite(str(images_dir / img_file.name), img)
            
            # 复制标注文件（YOLO格式是归一化的，不受图片尺寸影响）
            shutil.copy(label_file, labels_dir / label_file.name)
        
        if resized_count > 0:
            print(f"  ℹ {subset_name}: {resized_count}/{len(pairs)} 张图片已缩放以优化性能")
        
        print(f"✓ {subset_name} 数据复制完成")
    
    def create_yaml_config(self, class_names):
        """创建YOLO配置文件"""
        config = {
            'path': str(self.output_dir.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'nc': len(class_names),
            'names': class_names
        }
        
        yaml_path = self.output_dir / "data.yaml"
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, allow_unicode=True, sort_keys=False)
        
        print(f"✓ 配置文件已创建: {yaml_path}")
        return yaml_path
    
    def prepare(self, class_names):
        """执行完整的数据集准备流程"""
        print("=" * 60)
        print("YOLO数据集准备工具")
        print("=" * 60)
        
        # 1. 创建目录
        self.prepare_directories()
        
        # 2. 获取图片-标注对
        pairs = self.get_image_label_pairs()
        if not pairs:
            print("✗ 错误: 没有找到已标注的数据")
            return
        
        # 3. 划分数据集
        train_pairs, val_pairs = self.split_dataset(pairs)
        
        # 4. 复制文件
        self.copy_files(train_pairs, "train")
        self.copy_files(val_pairs, "val")
        
        # 5. 创建配置文件
        yaml_path = self.create_yaml_config(class_names)
        
        print("=" * 60)
        print("✓ 数据集准备完成！")
        print(f"配置文件: {yaml_path}")
        print(f"训练集: {len(train_pairs)} 张")
        print(f"验证集: {len(val_pairs)} 张")
        print("\n下一步：使用以下命令训练模型")
        print(f"  yolo detect train data={yaml_path} model=yolov8n.pt epochs=100")
        print("=" * 60)


def main():
    """主函数"""
    # 定义类别名称（根据你的模块类型修改）
    class_names = [
        'line_module',           # 0: 线路模块
        'button_module',         # 1: 按钮模块
        'four_color_block',      # 2: 四色方块模块
        'memory_module',         # 3: 记忆模块
        'maze_module',           # 4: 迷宫模块
        'serial_number',         # 5: 序列号区域
        'battery_holder',        # 6: 电池仓
        'indicator_light',       # 7: 指示灯
    ]
    
    preparer = YOLODatasetPreparer()
    preparer.prepare(class_names)


if __name__ == "__main__":
    main()
