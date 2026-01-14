"""
GPU检查工具 - 检测CUDA和GPU是否可用
"""
import sys


def check_torch_gpu():
    """检查PyTorch的GPU支持"""
    print("=" * 60)
    print("PyTorch GPU 检查")
    print("=" * 60)
    
    try:
        import torch
        print(f"✓ PyTorch 版本: {torch.__version__}")
        
        # CUDA检查
        cuda_available = torch.cuda.is_available()
        print(f"CUDA 可用: {'✓ 是' if cuda_available else '✗ 否'}")
        
        if cuda_available:
            print(f"CUDA 版本: {torch.version.cuda}")
            print(f"cuDNN 版本: {torch.backends.cudnn.version()}")
            print(f"GPU 数量: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
                print(f"  计算能力: {props.major}.{props.minor}")
                print(f"  总显存: {props.total_memory / 1024**3:.2f} GB")
                print(f"  多处理器数量: {props.multi_processor_count}")
                
                # 显存使用情况
                if i == 0:  # 只检查第一块卡
                    memory_allocated = torch.cuda.memory_allocated(i) / 1024**3
                    memory_reserved = torch.cuda.memory_reserved(i) / 1024**3
                    print(f"  已分配显存: {memory_allocated:.2f} GB")
                    print(f"  已预留显存: {memory_reserved:.2f} GB")
            
            # 测试GPU
            print("\n执行GPU测试...")
            x = torch.rand(1000, 1000, device='cuda')
            y = torch.rand(1000, 1000, device='cuda')
            z = torch.matmul(x, y)
            print("✓ GPU 运算测试通过")
            
        else:
            print("\n⚠ GPU不可用，可能原因：")
            print("1. 未安装支持CUDA的PyTorch版本")
            print("2. 未安装CUDA驱动或版本不匹配")
            print("3. 使用的是CPU版本的PyTorch")
            
            print("\n解决方案：")
            print("卸载现有PyTorch：")
            print("  pip uninstall torch torchvision torchaudio")
            print("\n安装GPU版本（CUDA 11.8）：")
            print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
            print("\n或CUDA 12.1版本：")
            print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
            
    except ImportError:
        print("✗ 未安装 PyTorch")
        print("\n安装命令：")
        print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")


def check_nvidia_driver():
    """检查NVIDIA驱动"""
    print("\n" + "=" * 60)
    print("NVIDIA 驱动检查")
    print("=" * 60)
    
    import subprocess
    try:
        result = subprocess.run(['nvidia-smi'], 
                              capture_output=True, 
                              text=True, 
                              timeout=5)
        if result.returncode == 0:
            print("✓ NVIDIA 驱动已安装\n")
            print(result.stdout)
        else:
            print("✗ nvidia-smi 执行失败")
    except FileNotFoundError:
        print("✗ 未找到 nvidia-smi 命令")
        print("请确保已安装NVIDIA驱动程序")
        print("下载地址: https://www.nvidia.com/Download/index.aspx")
    except Exception as e:
        print(f"✗ 检查失败: {e}")


def check_ultralytics():
    """检查Ultralytics YOLO"""
    print("\n" + "=" * 60)
    print("Ultralytics YOLO 检查")
    print("=" * 60)
    
    try:
        from ultralytics import YOLO, checks
        import ultralytics
        
        print(f"✓ Ultralytics 版本: {ultralytics.__version__}")
        
        # 运行环境检查
        print("\n运行环境检查...")
        checks()
        
    except ImportError:
        print("✗ 未安装 Ultralytics")
        print("\n安装命令：")
        print("  pip install ultralytics")


def main():
    """主函数"""
    print("\n🔍 GPU 环境完整检查\n")
    
    # 1. 检查NVIDIA驱动
    check_nvidia_driver()
    
    # 2. 检查PyTorch GPU支持
    check_torch_gpu()
    
    # 3. 检查Ultralytics
    check_ultralytics()
    
    print("\n" + "=" * 60)
    print("检查完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
