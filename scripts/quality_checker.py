"""
批量标注质量检查工具
用于检测标注错误和不一致
"""
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import json


class AnnotationQualityChecker:
    """标注质量检查器"""
    
    def __init__(self, images_dir: str, labels_dir: str):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        
        self.issues = []  # 记录发现的问题
        
    def check_all(self) -> Dict:
        """执行所有检查"""
        print("开始标注质量检查...")
        
        results = {
            'total_files': 0,
            'missing_labels': [],
            'missing_images': [],
            'empty_labels': [],
            'invalid_format': [],
            'out_of_bounds': [],
            'too_small': [],
            'too_large': [],
            'overlapping': [],
        }
        
        # 获取所有图片和标注
        image_files = list(self.images_dir.glob("*.jpg")) + \
                     list(self.images_dir.glob("*.png"))
        label_files = list(self.labels_dir.glob("*.txt"))
        
        image_stems = {f.stem for f in image_files}
        label_stems = {f.stem for f in label_files}
        
        results['total_files'] = len(image_files)
        
        # 检查1：缺失的标注
        results['missing_labels'] = list(image_stems - label_stems)
        
        # 检查2：缺失的图片
        results['missing_images'] = list(label_stems - image_stems)
        
        # 检查每个标注文件
        for label_file in label_files:
            # 检查对应图片是否存在
            img_path = None
            for ext in ['.jpg', '.png', '.jpeg']:
                temp = self.images_dir / f"{label_file.stem}{ext}"
                if temp.exists():
                    img_path = temp
                    break
            
            if not img_path:
                continue
            
            # 读取标注
            with open(label_file, 'r') as f:
                lines = f.readlines()
            
            # 检查3：空标注文件
            if not lines:
                results['empty_labels'].append(label_file.name)
                continue
            
            # 获取图片尺寸
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            img_h, img_w = img.shape[:2]
            
            # 解析标注
            boxes = []
            for line_num, line in enumerate(lines, 1):
                parts = line.strip().split()
                
                # 检查4：格式错误
                if len(parts) < 5:
                    results['invalid_format'].append(
                        f"{label_file.name}:L{line_num}"
                    )
                    continue
                
                try:
                    cls_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    
                    # 检查5：超出边界
                    if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and
                           0 <= width <= 1 and 0 <= height <= 1):
                        results['out_of_bounds'].append(
                            f"{label_file.name}:L{line_num}"
                        )
                    
                    # 计算实际像素尺寸
                    actual_w = width * img_w
                    actual_h = height * img_h
                    
                    # 检查6：框太小 (可能是误标)
                    if actual_w < 10 or actual_h < 10:
                        results['too_small'].append(
                            f"{label_file.name}:L{line_num} ({actual_w:.1f}x{actual_h:.1f}px)"
                        )
                    
                    # 检查7：框太大 (可能覆盖了整个图片)
                    if width > 0.9 or height > 0.9:
                        results['too_large'].append(
                            f"{label_file.name}:L{line_num}"
                        )
                    
                    boxes.append((x_center, y_center, width, height))
                    
                except ValueError:
                    results['invalid_format'].append(
                        f"{label_file.name}:L{line_num}"
                    )
            
            # 检查8：重叠的框 (可能是重复标注)
            for i in range(len(boxes)):
                for j in range(i + 1, len(boxes)):
                    if self._boxes_overlap(boxes[i], boxes[j]):
                        results['overlapping'].append(
                            f"{label_file.name}:Box{i+1}&Box{j+1}"
                        )
        
        return results
    
    def _boxes_overlap(self, box1: Tuple, box2: Tuple, threshold: float = 0.5) -> bool:
        """检查两个框是否重叠（IoU > threshold）"""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # 计算左上和右下坐标
        l1, t1 = x1 - w1/2, y1 - h1/2
        r1, b1 = x1 + w1/2, y1 + h1/2
        
        l2, t2 = x2 - w2/2, y2 - h2/2
        r2, b2 = x2 + w2/2, y2 + h2/2
        
        # 计算交集
        inter_l = max(l1, l2)
        inter_t = max(t1, t2)
        inter_r = min(r1, r2)
        inter_b = min(b1, b2)
        
        if inter_r < inter_l or inter_b < inter_t:
            return False
        
        inter_area = (inter_r - inter_l) * (inter_b - inter_t)
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - inter_area
        
        iou = inter_area / union_area if union_area > 0 else 0
        return iou > threshold
    
    def print_report(self, results: Dict):
        """打印检查报告"""
        print("\n" + "="*60)
        print("标注质量检查报告")
        print("="*60)
        
        print(f"\n总文件数: {results['total_files']}")
        
        # 打印各类问题
        issues = [
            ('缺失标注', results['missing_labels']),
            ('缺失图片', results['missing_images']),
            ('空标注文件', results['empty_labels']),
            ('格式错误', results['invalid_format']),
            ('超出边界', results['out_of_bounds']),
            ('框太小', results['too_small']),
            ('框太大', results['too_large']),
            ('重叠框', results['overlapping']),
        ]
        
        total_issues = 0
        for name, items in issues:
            count = len(items)
            total_issues += count
            
            if count > 0:
                print(f"\n⚠️  {name}: {count} 个")
                # 显示前几个示例
                for item in items[:5]:
                    print(f"   - {item}")
                if count > 5:
                    print(f"   ... 还有 {count - 5} 个")
        
        if total_issues == 0:
            print("\n✅ 未发现问题！标注质量良好。")
        else:
            print(f"\n⚠️  共发现 {total_issues} 个问题")
        
        print("="*60)
        
        return total_issues
    
    def export_report(self, results: Dict, output_path: str):
        """导出报告到JSON文件"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n报告已保存到: {output_path}")


def main():
    """主函数"""
    import sys
    
    if len(sys.argv) > 2:
        images_dir = sys.argv[1]
        labels_dir = sys.argv[2]
    else:
        images_dir = "dataset/raw_images"
        labels_dir = "dataset/manual_annotations"
    
    print(f"图片目录: {images_dir}")
    print(f"标注目录: {labels_dir}")
    
    checker = AnnotationQualityChecker(images_dir, labels_dir)
    results = checker.check_all()
    total_issues = checker.print_report(results)
    
    # 导出报告
    checker.export_report(results, "annotation_quality_report.json")
    
    return 0 if total_issues == 0 else 1


if __name__ == "__main__":
    exit(main())
