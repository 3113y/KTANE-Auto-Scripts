"""
YOLO 标注可视化和修正工具
用于查看自动标注结果，并支持手动调整
"""
import cv2
import numpy as np
from pathlib import Path
import sys

class YOLOAnnotationViewer:
    def __init__(self, images_dir="dataset/raw_images", labels_dir="dataset/labels"):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.image_files = sorted(list(self.images_dir.glob("*.jpg")) + list(self.images_dir.glob("*.png")))
        self.current_idx = 0
        self.class_names = ["module"]  # 根据实际类别修改
        self.colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0)]
        
    def load_yolo_annotations(self, label_path, img_width, img_height):
        """加载 YOLO 格式标注"""
        if not label_path.exists():
            return []
        
        annotations = []
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    x_center = float(parts[1]) * img_width
                    y_center = float(parts[2]) * img_height
                    width = float(parts[3]) * img_width
                    height = float(parts[4]) * img_height
                    
                    x1 = int(x_center - width / 2)
                    y1 = int(y_center - height / 2)
                    x2 = int(x_center + width / 2)
                    y2 = int(y_center + height / 2)
                    
                    annotations.append((cls_id, x1, y1, x2, y2))
        return annotations
    
    def draw_annotations(self, image, annotations):
        """在图像上绘制标注框"""
        img_copy = image.copy()
        for cls_id, x1, y1, x2, y2 in annotations:
            color = self.colors[cls_id % len(self.colors)]
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)
            
            # 绘制类别标签
            label = f"{self.class_names[cls_id] if cls_id < len(self.class_names) else 'class_' + str(cls_id)}"
            cv2.putText(img_copy, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return img_copy
    
    def show_image(self):
        """显示当前图像及其标注"""
        if self.current_idx >= len(self.image_files):
            print("已到达最后一张图片")
            return None
            
        img_path = self.image_files[self.current_idx]
        image = cv2.imread(str(img_path))
        
        if image is None:
            print(f"无法加载图片: {img_path}")
            return None
        
        # 加载对应的标注文件
        label_path = self.labels_dir / (img_path.stem + ".txt")
        annotations = self.load_yolo_annotations(label_path, image.shape[1], image.shape[0])
        
        # 绘制标注
        img_with_boxes = self.draw_annotations(image, annotations)
        
        # 添加信息文本
        info_text = f"图片 {self.current_idx + 1}/{len(self.image_files)} | {img_path.name} | 标注数: {len(annotations)}"
        cv2.putText(img_with_boxes, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(img_with_boxes, "按键: A-上一张 | D-下一张 | Q-退出 | S-保存当前", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return img_with_boxes
    
    def run(self):
        """运行可视化程序"""
        print("YOLO 标注可视化工具")
        print("=" * 50)
        print("操作说明:")
        print("  A 键: 上一张图片")
        print("  D 键: 下一张图片")
        print("  Q 键: 退出程序")
        print("  S 键: 保存当前图片(带标注)")
        print("=" * 50)
        
        cv2.namedWindow("YOLO Annotations", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("YOLO Annotations", 1280, 720)
        
        while True:
            img_with_boxes = self.show_image()
            
            if img_with_boxes is None:
                break
            
            cv2.imshow("YOLO Annotations", img_with_boxes)
            
            key = cv2.waitKey(0) & 0xFF
            
            if key == ord('q') or key == 27:  # Q 或 ESC 退出
                break
            elif key == ord('d'):  # D 下一张
                self.current_idx = min(self.current_idx + 1, len(self.image_files) - 1)
            elif key == ord('a'):  # A 上一张
                self.current_idx = max(self.current_idx - 1, 0)
            elif key == ord('s'):  # S 保存
                output_dir = Path("dataset/annotated_preview")
                output_dir.mkdir(exist_ok=True)
                output_path = output_dir / self.image_files[self.current_idx].name
                cv2.imwrite(str(output_path), img_with_boxes)
                print(f"已保存: {output_path}")
        
        cv2.destroyAllWindows()
        print("\n程序已退出")

def main():
    viewer = YOLOAnnotationViewer()
    viewer.run()

if __name__ == "__main__":
    main()
