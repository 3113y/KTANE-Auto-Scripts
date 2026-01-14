"""
YOLO模型训练脚本 - 自动检测并使用GPU加速
"""
import torch
from ultralytics import YOLO
from pathlib import Path


def check_gpu():
    """检查GPU可用性"""
    print("=" * 60)
    print("系统环境检查")
    print("=" * 60)
    
    # 检查CUDA
    cuda_available = torch.cuda.is_available()
    print(f"CUDA 可用: {cuda_available}")
    
    if cuda_available:
        print(f"CUDA 版本: {torch.version.cuda}")
        print(f"GPU 数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  显存: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
    else:
        print("⚠ 警告: 未检测到GPU，将使用CPU训练（速度较慢）")
        print("\n如需使用GPU，请确保：")
        print("1. 安装了支持CUDA的PyTorch版本")
        print("2. 安装了对应版本的CUDA和cuDNN")
        print("3. 显卡驱动已更新")
        print("\n安装GPU版本PyTorch：")
        print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
    
    print("=" * 60)
    return cuda_available


def train_model(data_yaml="dataset/yolo/data.yaml",
                model_name="yolov8n.pt",
                epochs=100,
                imgsz=640,
                batch=16,
                device=None):
    """
    训练YOLO模型
    
    Args:
        data_yaml: 数据集配置文件路径
        model_name: 预训练模型名称（yolov8n.pt, yolov8s.pt, yolov8m.pt等）
        epochs: 训练轮数
        imgsz: 输入图片尺寸
        batch: 批次大小（如果显存不足会自动减小）
        device: 设备（None=自动选择, 0=GPU0, 'cpu'=CPU）
    """
    # 检查GPU
    gpu_available = check_gpu()
    
    # 自动选择设备
    if device is None:
        device = 0 if gpu_available else 'cpu'
    
    # 检查数据集配置文件
    data_path = Path(data_yaml)
    if not data_path.exists():
        print(f"✗ 错误: 数据集配置文件不存在: {data_path}")
        print("\n请先运行: python scripts/prepare_yolo_dataset.py")
        return
    
    print(f"\n开始训练配置")
    print(f"  数据集: {data_yaml}")
    print(f"  模型: {model_name}")
    print(f"  训练轮数: {epochs}")
    print(f"  图片尺寸: {imgsz}")
    print(f"  批次大小: {batch}")
    print(f"  设备: {device}")
    print("=" * 60)
    
    # 加载模型
    model = YOLO(model_name)
    
    # 开始训练
    try:
        results = model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,
            patience=50,  # 早停耐心值
            save=True,
            plots=True,
            # 优化设置
            optimizer='AdamW',
            lr0=0.01,
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3.0,
            # 数据增强
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=0.0,
            translate=0.1,
            scale=0.5,
            shear=0.0,
            perspective=0.0,
            flipud=0.0,
            fliplr=0.5,
            mosaic=1.0,
        )
        
        print("\n" + "=" * 60)
        print("✓ 训练完成！")
        print(f"模型已保存到: runs/detect/train/weights/best.pt")
        print("=" * 60)
        
        return results
        
    except RuntimeError as e:
        if "out of memory" in str(e):
            print("\n✗ 显存不足！建议：")
            print("1. 减小batch大小（当前: {batch}）")
            print("2. 减小图片尺寸（当前: {imgsz}）")
            print("3. 使用更小的模型（如yolov8n.pt）")
            print("\n重试命令示例：")
            print(f"  python scripts/train_yolo.py --batch 8 --imgsz 480")
        raise


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="YOLO模型训练")
    parser.add_argument("--data", default="dataset/yolo/data.yaml", help="数据集配置文件")
    parser.add_argument("--model", default="yolov8n.pt", help="预训练模型")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--imgsz", type=int, default=640, help="图片尺寸")
    parser.add_argument("--batch", type=int, default=16, help="批次大小")
    parser.add_argument("--device", default=None, help="设备（0=GPU, cpu=CPU）")
    
    args = parser.parse_args()
    
    # 训练模型
    train_model(
        data_yaml=args.data,
        model_name=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device
    )


if __name__ == "__main__":
    main()
