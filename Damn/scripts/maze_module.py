from PyQt5.QtWidgets import QInputDialog, QMessageBox, QGridLayout, QPushButton, QLabel, QVBoxLayout
from PyQt5.QtGui import QPixmap
import os

def maze_module(parent):
    parent.mark_site = [0, 0]
    parent.maze_click_count = 0
    parent.first_col = 0
    parent.second_col = 0
    grid = QGridLayout()
    for row in range(6):
        for col in range(6):
            button = QPushButton(f'({row+1}, {col+1})')
            button.clicked.connect(lambda _, r=row, c=col: on_button_click(parent, r, c))
            grid.addWidget(button, row, col)
    parent.setLayout(grid)
    parent.setWindowTitle('选择两个标识方块的位置')

def on_button_click(parent, row, col):
    if parent.maze_click_count == 0:
        parent.mark_site[0] = row + 1
        parent.first_col = col + 1
    else:
        parent.mark_site[1] = row + 1
        parent.second_col = col + 1

    parent.maze_click_count += 1
    if parent.maze_click_count == 2:
        pic_name = parent.mark_site[0] * 10 + parent.first_col
        file_name = f"pictures\\{pic_name}.jpg"
        parent.file_name = file_name
        load_and_display_image(parent)

def load_and_display_image(parent):
    if not os.path.exists(parent.file_name):
        parent.switch_module()
        return
    pixmap = QPixmap(parent.file_name)
    if pixmap.isNull():
        parent.switch_module()
        return
    layout = QVBoxLayout()
    image_label = QLabel(parent)
    image_label.setPixmap(pixmap)
    layout.addWidget(image_label)
    return_button = QPushButton("返回主菜单", parent)
    return_button.clicked.connect(parent.switch_module)
    layout.addWidget(return_button)
    parent.setLayout(layout)
    parent.resize(pixmap.width(), pixmap.height())