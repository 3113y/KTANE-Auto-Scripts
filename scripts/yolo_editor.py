"""
YOLO 标注编辑器 - 支持查看、添加、删除、调整标注框
"""
import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

class YOLOEditor:
    def __init__(self, images_dir="dataset/raw_images", labels_dir="dataset/labels"):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.labels_dir.mkdir(exist_ok=True)
        
        self.image_files = sorted(list(self.images_dir.glob("*.jpg")) + list(self.images_dir.glob("*.png")))
        self.current_idx = 0
        
        # 标注数据
        self.annotations = []  # [(cls_id, x1, y1, x2, y2), ...]
        self.current_image = None
        self.original_image = None
        
        # 显示尺寸设置（用于优化性能，适配 YOLO 训练）
        self.max_display_width = 960
        self.max_display_height = 960
        self.scale_factor = 1.0  # 缩放因子
        
        # 鼠标交互状态
        self.drawing = False
        self.start_point = None
        self.current_box = None
        self.selected_box_idx = None
        self.dragging = False
        
        # 当前类别
        self.current_class = 0
        self.class_names = [
            "port", "serial", "indicator", "battery", "knob", "capacitor",
            "vent", "password", "seq_wire", "complex_wire", "morse",
            "maze", "memory", "who", "color_block", "keypad", "button", "wire"
        ]
        self.class_names_cn = [
            "端口表", "序列号", "指示表", "电池表", "旋钮", "电容",
            "排气", "密码", "顺序线路", "复杂线路", "莫斯电码",
            "迷宫", "记忆", "叫什么", "四色方块", "键盘", "按钮", "线路"
        ]
        # 为不同类别分配不同颜色
        self.colors = [
            (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
            (0, 255, 255), (128, 255, 0), (255, 128, 0), (128, 0, 255), (255, 0, 128),
            (0, 128, 255), (128, 255, 128), (255, 128, 128), (128, 128, 255), (255, 255, 128),
            (128, 255, 255), (255, 128, 255), (200, 200, 200)
        ]
        
        # 中文字体 (使用系统字体，增大字号)
        self.font = None
        try:
            # Windows 系统字体路径
            font_path = "C:/Windows/Fonts/msyh.ttc"  # 微软雅黑
            self.font = ImageFont.truetype(font_path, 28)  # 增大到28
            self.font_small = ImageFont.truetype(font_path, 20)  # 增大到20
            self.font_tiny = ImageFont.truetype(font_path, 16)  # 列表用
        except:
            print("警告: 无法加载中文字体，将使用英文标签")
            self.font = None
        
        # 显示设置
        self.show_help = True
        
    def cv2_add_chinese_text(self, img, text, position, font, color):
        """在 OpenCV 图像上添加中文文本"""
        if self.font is None:
            # 回退到英文
            cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            return img
        
        # 转换为 PIL 格式
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        
        # 绘制文本
        draw.text(position, text, font=font, fill=color[::-1])  # BGR to RGB
        
        # 转回 OpenCV 格式
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    
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
                    
                    annotations.append([cls_id, x1, y1, x2, y2])
        return annotations
    
    def save_yolo_annotations(self, label_path, annotations, img_width, img_height):
        """保存 YOLO 格式标注（需要转回原始尺寸）"""
        with open(label_path, 'w') as f:
            for cls_id, x1, y1, x2, y2 in annotations:
                # 转回原始尺寸
                x1_orig = x1 / self.scale_factor
                y1_orig = y1 / self.scale_factor
                x2_orig = x2 / self.scale_factor
                y2_orig = y2 / self.scale_factor
                
                # 计算原始尺寸的中心点和宽高（归一化）
                orig_w = img_width / self.scale_factor
                orig_h = img_height / self.scale_factor
                
                x_center = ((x1_orig + x2_orig) / 2) / orig_w
                y_center = ((y1_orig + y2_orig) / 2) / orig_h
                width = (x2_orig - x1_orig) / orig_w
                height = (y2_orig - y1_orig) / orig_h
                
                f.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
    
    def load_image(self):
        """加载当前图像和标注"""
        if self.current_idx >= len(self.image_files):
            return False
        
        img_path = self.image_files[self.current_idx]
        original = cv2.imread(str(img_path))
        
        if original is None:
            print(f"无法加载: {img_path}")
            return False
        
        # 计算缩放因子以优化性能
        h, w = original.shape[:2]
        scale_w = self.max_display_width / w
        scale_h = self.max_display_height / h
        self.scale_factor = min(1.0, scale_w, scale_h)  # 不放大，只缩小
        
        # 如果需要缩放则缩放图像
        if self.scale_factor < 1.0:
            new_w = int(w * self.scale_factor)
            new_h = int(h * self.scale_factor)
            self.original_image = cv2.resize(original, (new_w, new_h), interpolation=cv2.INTER_AREA)
            print(f"图片已从 {w}x{h} 缩放至 {new_w}x{new_h} (缩放比例: {self.scale_factor:.2f})")
        else:
            self.original_image = original
        
        # 加载标注（使用原始尺寸）
        label_path = self.labels_dir / (img_path.stem + ".txt")
        annotations_original = self.load_yolo_annotations(label_path, w, h)
        
        # 将标注缩放到显示尺寸
        self.annotations = []
        for cls_id, x1, y1, x2, y2 in annotations_original:
            self.annotations.append([
                cls_id,
                int(x1 * self.scale_factor),
                int(y1 * self.scale_factor),
                int(x2 * self.scale_factor),
                int(y2 * self.scale_factor)
            ])
        
        self.selected_box_idx = None
        return True
    
    def draw_display(self):
        """绘制显示图像"""
        self.current_image = self.original_image.copy()
        h, w = self.current_image.shape[:2]
        
        # 绘制所有标注框
        for idx, (cls_id, x1, y1, x2, y2) in enumerate(self.annotations):
            color = self.colors[cls_id % len(self.colors)]
            thickness = 3 if idx == self.selected_box_idx else 2
            cv2.rectangle(self.current_image, (x1, y1), (x2, y2), color, thickness)
            
            # 标签（使用中文）
            if cls_id < len(self.class_names_cn):
                label = self.class_names_cn[cls_id]
            else:
                label = f'cls_{cls_id}'
            
            if self.font:
                self.current_image = self.cv2_add_chinese_text(
                    self.current_image, label, (x1, max(y1 - 25, 0)), self.font_small, color
                )
            else:
                cv2.putText(self.current_image, self.class_names[cls_id], (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # 绘制正在绘制的框
        if self.drawing and self.start_point and self.current_box:
            color = self.colors[self.current_class % len(self.colors)]
            cv2.rectangle(self.current_image, self.start_point, self.current_box, color, 2)
        
        # 信息栏 - 增大字体
        img_name = self.image_files[self.current_idx].name
        info = f"[{self.current_idx + 1}/{len(self.image_files)}] {img_name} | Boxes: {len(self.annotations)}"
        cv2.putText(self.current_image, info, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # 当前类别高亮显示（中文）- 增大字体
        current_cls_cn = self.class_names_cn[self.current_class] if self.current_class < len(self.class_names_cn) else f"cls_{self.current_class}"
        cls_info = f"Class: [{self.current_class}] {current_cls_cn}"
        
        if self.font:
            self.current_image = self.cv2_add_chinese_text(
                self.current_image, cls_info, (10, 70), self.font, 
                self.colors[self.current_class % len(self.colors)]
            )
        else:
            cv2.putText(self.current_image, f"Class: [{self.current_class}] {self.class_names[self.current_class]}", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 
                       self.colors[self.current_class % len(self.colors)], 2)
        
        # 类别列表（右侧显示）- 支持滚动显示
        list_x = w - 200
        list_y = 30
        
        # 计算列表高度和可显示数量
        max_visible = min(10, len(self.class_names_cn))  # 最多显示10个
        item_height = 28
        list_height = max_visible * item_height + 35
        
        cv2.rectangle(self.current_image, (list_x - 10, list_y - 25), 
                     (w - 10, list_y + list_height), (40, 40, 40), -1)
        cv2.putText(self.current_image, "Classes:", (list_x, list_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 计算滚动偏移，确保当前类别可见
        start_idx = max(0, self.current_class - max_visible // 2)
        end_idx = min(len(self.class_names_cn), start_idx + max_visible)
        if end_idx == len(self.class_names_cn):
            start_idx = max(0, end_idx - max_visible)
        
        if self.font:
            for i, idx in enumerate(range(start_idx, end_idx)):
                cls_name_cn = self.class_names_cn[idx]
                y_pos = list_y + (i + 1) * item_height
                color = self.colors[idx % len(self.colors)]
                prefix = ">> " if idx == self.current_class else "   "
                text = f"{prefix}[{idx}] {cls_name_cn}"
                self.current_image = self.cv2_add_chinese_text(
                    self.current_image, text, (list_x, y_pos), self.font_tiny, color
                )
        else:
            for i, idx in enumerate(range(start_idx, end_idx)):
                cls_name = self.class_names[idx]
                y_pos = list_y + (i + 1) * item_height
                color = self.colors[idx % len(self.colors)]
                prefix = ">> " if idx == self.current_class else "   "
                text = f"{prefix}[{idx}] {cls_name}"
                thickness = 2 if idx == self.current_class else 1
                cv2.putText(self.current_image, text, (list_x, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)
        
        # 帮助信息 - 增大字体
        if self.show_help:
            help_text = [
                "Mouse: Drag-New | Right-Del | A/D: Prev/Next",
                "S: Save | C: Switch Class | H: Hide | Q/ESC: Quit"
            ]
            y_offset = h - 50
            for text in help_text:
                cv2.putText(self.current_image, text, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                y_offset += 25
    
    def find_box_at_point(self, x, y):
        """查找点击位置的标注框"""
        for idx, (cls_id, x1, y1, x2, y2) in enumerate(self.annotations):
            if x1 <= x <= x2 and y1 <= y <= y2:
                return idx
        return None
    
    def mouse_callback(self, event, x, y, flags, param):
        """鼠标事件处理"""
        if event == cv2.EVENT_LBUTTONDOWN:
            # 检查是否点击在已有框上
            clicked_idx = self.find_box_at_point(x, y)
            if clicked_idx is not None:
                self.selected_box_idx = clicked_idx
                self.dragging = True
            else:
                # 开始绘制新框
                self.drawing = True
                self.start_point = (x, y)
                self.current_box = (x, y)
                self.selected_box_idx = None
        
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.current_box = (x, y)
        
        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing:
                # 完成绘制
                self.drawing = False
                x1, y1 = self.start_point
                x2, y2 = x, y
                
                # 确保坐标顺序正确
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)
                
                # 过滤太小的框
                if (x2 - x1) > 10 and (y2 - y1) > 10:
                    self.annotations.append([self.current_class, x1, y1, x2, y2])
                    print(f"添加标注框: 类别{self.current_class}, ({x1},{y1}) -> ({x2},{y2})")
                
                self.start_point = None
                self.current_box = None
            self.dragging = False
        
        elif event == cv2.EVENT_RBUTTONDOWN:
            # 右键删除框
            clicked_idx = self.find_box_at_point(x, y)
            if clicked_idx is not None:
                deleted = self.annotations.pop(clicked_idx)
                print(f"删除标注框: {deleted}")
                self.selected_box_idx = None
    
    def save_current(self):
        """保存当前图像的标注"""
        img_path = self.image_files[self.current_idx]
        label_path = self.labels_dir / (img_path.stem + ".txt")
        self.save_yolo_annotations(
            label_path,
            self.annotations,
            self.original_image.shape[1],
            self.original_image.shape[0]
        )
        print(f"已保存: {label_path} ({len(self.annotations)} 个标注)")
    
    def run(self):
        """运行编辑器"""
        print("=" * 60)
        print("YOLO 标注编辑器")
        print("=" * 60)
        print("操作说明:")
        print("  鼠标左键拖动: 绘制新标注框")
        print("  鼠标右键: 删除点击的标注框")
        print("  A 键: 上一张图片")
        print("  D 键: 下一张图片")
        print("  S 键: 保存当前标注")
        print("  C 键: 切换标注类别")
        print("  H 键: 显示/隐藏帮助信息")
        print("  Q 键/ESC: 退出编辑器")
        print("=" * 60)
        
        if not self.load_image():
            print("无图片可加载")
            return
        
        window_name = "YOLO Editor"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720)
        cv2.setMouseCallback(window_name, self.mouse_callback)
        
        while True:
            self.draw_display()
            cv2.imshow(window_name, self.current_image)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27:  # Q 或 ESC
                # 询问是否保存
                print("\n是否保存当前修改? (Y/n)")
                break
            elif key == ord('a'):  # 上一张
                self.save_current()
                self.current_idx = max(0, self.current_idx - 1)
                self.load_image()
            elif key == ord('d'):  # 下一张
                self.save_current()
                self.current_idx = min(len(self.image_files) - 1, self.current_idx + 1)
                self.load_image()
            elif key == ord('s'):  # 保存
                self.save_current()
            elif key == ord('c'):  # 切换类别
                self.current_class = (self.current_class + 1) % len(self.class_names)
                cls_display = self.class_names_cn[self.current_class] if self.current_class < len(self.class_names_cn) else self.class_names[self.current_class]
                print(f"切换到类别: [{self.current_class}] {cls_display}")
            elif key == ord('h'):  # 切换帮助显示
                self.show_help = not self.show_help
        
        cv2.destroyAllWindows()
        print("\n编辑器已关闭")

def main():
    editor = YOLOEditor()
    editor.run()

if __name__ == "__main__":
    main()
