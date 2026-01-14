"""
半自动标注辅助工具 - 使用传统CV方法预标注，减少人工工作量
"""
import cv2
import numpy as np
from pathlib import Path
import json

class AutoAnnotateHelper:
    def __init__(self, images_dir="dataset/raw_images"):
        """初始化自动标注辅助器"""
        self.images_dir = Path(images_dir)
        self.annotations = []
    
    def detect_modules_by_color(self, image):
        """
        使用颜色和轮廓检测可能的模块区域
        返回检测到的边界框列表
        """
        # 转换到HSV空间便于颜色检测
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 定义模块可能的颜色范围（需要根据实际游戏界面调整）
        # 这里假设模块区域有明显的边框或背景色
        lower_bound = np.array([0, 0, 100])  # 示例：亮色区域
        upper_bound = np.array([180, 50, 255])
        
        mask = cv2.inRange(hsv, lower_bound, upper_bound)
        
        # 形态学处理
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # 查找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        bboxes = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 5000:  # 过滤小区域
                x, y, w, h = cv2.boundingRect(contour)
                bboxes.append((x, y, w, h))
        
        return bboxes
    
    def detect_by_template(self, image, template_path):
        """
        模板匹配检测（适用于固定样式的UI元素）
        """
        if not Path(template_path).exists():
            return []
        
        template = cv2.imread(template_path, 0)
        if template is None:
            return []
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
        threshold = 0.7
        locations = np.where(result >= threshold)
        
        bboxes = []
        h, w = template.shape
        for pt in zip(*locations[::-1]):
            bboxes.append((pt[0], pt[1], w, h))
        
        # 使用非极大值抑制去除重叠的检测框
        bboxes = self._non_max_suppression(bboxes, overlap_thresh=0.3)
        
        return bboxes
    
    def _non_max_suppression(self, bboxes, overlap_thresh=0.3):
        """
        非极大值抑制 - 去除重叠的边界框
        """
        if len(bboxes) == 0:
            return []
        
        # 转换为 (x1, y1, x2, y2) 格式
        boxes = np.array([(x, y, x+w, y+h) for x, y, w, h in bboxes])
        
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        indices = np.arange(len(boxes))
        
        selected = []
        
        while len(indices) > 0:
            # 选择第一个框
            i = indices[0]
            selected.append(i)
            
            # 计算与其他框的交集
            xx1 = np.maximum(x1[i], x1[indices[1:]])
            yy1 = np.maximum(y1[i], y1[indices[1:]])
            xx2 = np.minimum(x2[i], x2[indices[1:]])
            yy2 = np.minimum(y2[i], y2[indices[1:]])
            
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)
            
            intersection = w * h
            union = areas[i] + areas[indices[1:]] - intersection
            iou = intersection / union
            
            # 保留IoU小于阈值的框
            indices = indices[1:][iou <= overlap_thresh]
        
        # 转换回 (x, y, w, h) 格式
        result = []
        for i in selected:
            x, y, w, h = bboxes[i]
            result.append((x, y, w, h))
        
        return result
    
    def visualize_detections(self, image, bboxes, class_id=0):
        """可视化检测结果"""
        vis_image = image.copy()
        colors = [
            (0, 255, 0),    # 绿色
            (255, 0, 0),    # 蓝色
            (0, 0, 255),    # 红色
            (255, 255, 0),  # 青色
            (255, 0, 255),  # 品红
        ]
        
        for x, y, w, h in bboxes:
            color = colors[class_id % len(colors)]
            cv2.rectangle(vis_image, (x, y), (x+w, y+h), color, 2)
            cv2.putText(vis_image, f"Class {class_id}", (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return vis_image
    
    def bbox_to_yolo_format(self, bbox, img_width, img_height):
        """
        将边界框转换为YOLO格式
        bbox: (x, y, w, h) - 左上角坐标和宽高
        返回: (x_center, y_center, width, height) - 归一化的中心坐标和宽高
        """
        x, y, w, h = bbox
        x_center = (x + w/2) / img_width
        y_center = (y + h/2) / img_height
        width = w / img_width
        height = h / img_height
        return (x_center, y_center, width, height)
    
    def save_annotation(self, image_path, bboxes, class_ids, output_dir="dataset/labels"):
        """保存YOLO格式的标注文件"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"✗ 无法读取图片: {image_path}")
            return
        img_height, img_width = img.shape[:2]
        
        # 生成标注文件
        label_path = output_dir / f"{Path(image_path).stem}.txt"
        
        with open(label_path, 'w') as f:
            for bbox, class_id in zip(bboxes, class_ids):
                x_center, y_center, width, height = self.bbox_to_yolo_format(
                    bbox, img_width, img_height
                )
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
        
        print(f"✓ 保存标注: {label_path.name}")
    
    def process_directory(self, visualize=True):
        """批量处理目录中的所有图片"""
        image_extensions = ['.png', '.jpg', '.jpeg']
        image_files = [f for f in self.images_dir.iterdir() 
                      if f.suffix.lower() in image_extensions]
        
        print(f"找到 {len(image_files)} 张图片")
        
        for img_path in image_files:
            print(f"\n处理: {img_path.name}")
            image = cv2.imread(str(img_path))
            
            # 自动检测
            bboxes = self.detect_modules_by_color(image)
            
            if bboxes:
                print(f"  检测到 {len(bboxes)} 个候选区域")
                
                # 假设所有检测到的都是同一类别（实际应用中需要分类）
                class_ids = [0] * len(bboxes)
                
                # 保存标注（预标注）
                self.save_annotation(img_path, bboxes, class_ids)
                
                # 可视化
                if visualize:
                    vis_image = self.visualize_detections(image, bboxes)
                    cv2.imshow("检测结果 (按任意键继续)", vis_image)
                    cv2.waitKey(500)
            else:
                print("  未检测到区域")
        
        cv2.destroyAllWindows()
        print("\n✓ 预标注完成！请使用LabelImg检查和修正标注")


def main():
    """主函数"""
    print("=" * 60)
    print("半自动标注辅助工具")
    print("此工具会生成初步标注，请务必用LabelImg检查修正！")
    print("=" * 60)
    
    helper = AutoAnnotateHelper()
    helper.process_directory(visualize=True)
    
    print("\n下一步:")
    print("1. 使用LabelImg打开 dataset/raw_images 目录")
    print("2. 检查并修正自动生成的标注")
    print("3. 标注完成后运行 prepare_yolo_dataset.py")


if __name__ == "__main__":
    main()
