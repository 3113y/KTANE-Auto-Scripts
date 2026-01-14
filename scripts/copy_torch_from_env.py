"""
从其他Python环境复制GPU版PyTorch到当前环境
"""
import os
import sys
import subprocess
import shutil
from pathlib import Path


def find_python_environments():
    """查找常见Python环境位置"""
    possible_paths = [
        Path.home() / ".conda" / "envs",
        Path.home() / "anaconda3" / "envs",
        Path.home() / "miniconda3" / "envs",
        Path("C:/ProgramData/Anaconda3/envs"),
        Path("C:/Users") / os.getlogin() / "AppData/Local/Programs/Python",
        Path("C:/Python39"),
        Path("C:/Python310"),
        Path("C:/Python311"),
        Path("C:/Python312"),
    ]
    
    envs = []
    for base_path in possible_paths:
        if base_path.exists():
            if "envs" in str(base_path):
                # Conda环境目录
                for env_dir in base_path.iterdir():
                    if env_dir.is_dir():
                        python_exe = env_dir / "python.exe"
                        if python_exe.exists():
                            envs.append(env_dir)
            else:
                # 直接Python安装
                python_exe = base_path / "python.exe"
                if python_exe.exists():
                    envs.append(base_path)
    
    return envs


def check_torch_version(python_path):
    """检查Python环境中的torch版本"""
    try:
        result = subprocess.run(
            [str(python_path), "-c", "import torch; print(torch.__version__, torch.cuda.is_available())"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            output = result.stdout.strip()
            if "True" in output:  # CUDA可用
                return output
    except:
        pass
    return None


def list_available_envs():
    """列出所有可用的Python环境及其torch版本"""
    print("=" * 80)
    print("搜索其他Python环境中的GPU版PyTorch...")
    print("=" * 80)
    
    envs = find_python_environments()
    gpu_envs = []
    
    for i, env_path in enumerate(envs, 1):
        python_exe = env_path / "python.exe"
        torch_info = check_torch_version(python_exe)
        
        if torch_info:
            print(f"\n✓ 找到GPU环境 [{i}]:")
            print(f"  路径: {env_path}")
            print(f"  PyTorch: {torch_info}")
            gpu_envs.append((i, env_path, python_exe))
    
    if not gpu_envs:
        print("\n✗ 未找到包含GPU版PyTorch的环境")
        print("\n请手动指定环境路径，或重新下载安装")
    
    return gpu_envs


def copy_torch_packages(source_python, target_env):
    """复制torch相关包到目标环境"""
    print("\n" + "=" * 80)
    print("开始复制PyTorch包...")
    print("=" * 80)
    
    # 获取源环境的site-packages路径
    result = subprocess.run(
        [str(source_python), "-c", 
         "import site; print(site.getsitepackages()[0])"],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print("✗ 无法获取源环境的site-packages路径")
        print(f"错误: {result.stderr}")
        return False
    
    source_site = Path(result.stdout.strip())
    
    # 目标环境的site-packages
    if isinstance(target_env, str):
        target_env = Path(target_env)
    target_site = target_env / "Lib" / "site-packages"
    
    if not source_site.exists():
        print(f"✗ 源环境site-packages不存在: {source_site}")
        return False
    
    if not target_site.exists():
        print(f"✗ 目标环境site-packages不存在: {target_site}")
        return False
    
    print(f"\n源路径: {source_site}")
    print(f"目标路径: {target_site}\n")
    
    # 需要复制的包
    packages = [
        "torch",
        "torch-*.dist-info",
        "torchvision", 
        "torchvision-*.dist-info",
        "torchaudio",
        "torchaudio-*.dist-info",
        "nvidia",
        "nvfuser",
        "nvtx",
        "triton",
        "filelock-*.dist-info",
        "filelock.py",
        "mpmath",
        "mpmath-*.dist-info",
        "networkx",
        "networkx-*.dist-info",
        "sympy",
        "sympy-*.dist-info",
        "fsspec",
        "fsspec-*.dist-info",
    ]
    
    copied_count = 0
    total_size = 0
    
    for pattern in packages:
        matching_items = list(source_site.glob(pattern))
        
        for item in matching_items:
            target_item = target_site / item.name
            
            try:
                if item.is_dir():
                    if target_item.exists():
                        shutil.rmtree(target_item)
                    shutil.copytree(item, target_item)
                    print(f"  ✓ 复制目录: {item.name}")
                else:
                    shutil.copy2(item, target_item)
                    print(f"  ✓ 复制文件: {item.name}")
                
                copied_count += 1
                
                # 计算大小
                if item.is_dir():
                    size = sum(f.stat().st_size for f in item.rglob('*') if f.is_file())
                else:
                    size = item.stat().st_size
                total_size += size
                
            except Exception as e:
                print(f"  ⚠ 复制失败 {item.name}: {e}")
    
    print(f"\n✓ 完成！复制了 {copied_count} 个项目，总计 {total_size / 1024**3:.2f} GB")
    return True


def verify_installation(target_python):
    """验证安装"""
    print("\n" + "=" * 80)
    print("验证安装...")
    print("=" * 80)
    
    result = subprocess.run(
        [str(target_python), "-c", 
         "import torch; print(f'PyTorch {torch.__version__}'); "
         "print(f'CUDA available: {torch.cuda.is_available()}'); "
         "print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print(result.stdout)
        if "True" in result.stdout:
            print("\n✓ GPU版PyTorch安装成功！")
            return True
    
    print("\n✗ 验证失败")
    print(result.stderr if result.stderr else result.stdout)
    return False


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="从其他环境复制GPU版PyTorch")
    parser.add_argument("--source", help="源环境路径（如: D:\\Anaconda\\envs\\yolov11）")
    args = parser.parse_args()
    
    print("\n🔄 PyTorch GPU版本复制工具\n")
    
    # 当前虚拟环境
    current_env = Path("E:/works/DAMN/.venv")
    current_python = current_env / "Scripts" / "python.exe"
    
    if not current_python.exists():
        print(f"✗ 当前环境Python不存在: {current_python}")
        return
    
    print(f"目标环境: {current_env}\n")
    
    # 如果指定了源路径，直接使用
    if args.source:
        source_env = Path(args.source)
        if not source_env.exists():
            print(f"✗ 源环境不存在: {source_env}")
            return
        
        source_python = source_env / "python.exe"
        if not source_python.exists():
            print(f"✗ 源环境Python不存在: {source_python}")
            return
        
        # 检查是否包含GPU版PyTorch
        torch_info = check_torch_version(source_python)
        if not torch_info:
            print(f"⚠ 警告: 源环境可能不包含GPU版PyTorch")
            confirm = input("是否继续复制？ (yes/no): ").strip()
            if confirm.lower() != 'yes':
                print("已取消")
                return
        else:
            print(f"✓ 源环境PyTorch: {torch_info}\n")
        
        if copy_torch_packages(source_python, current_env):
            verify_installation(current_python)
        return
    
    # 否则查找可用环境
    gpu_envs = list_available_envs()
    
    if gpu_envs:
        print("\n" + "=" * 80)
        choice = input(f"\n请选择要复制的环境编号 [1-{len(gpu_envs)}] (或输入 'q' 退出): ").strip()
        
        if choice.lower() == 'q':
            print("已取消")
            return
        
        try:
            idx = int(choice)
            if 1 <= idx <= len(gpu_envs):
                selected = gpu_envs[idx - 1]
                source_python = selected[2]
                
                confirm = input(f"\n确认从以下环境复制？\n  {selected[1]}\n\n输入 'yes' 确认: ").strip()
                
                if confirm.lower() == 'yes':
                    if copy_torch_packages(source_python, current_env):
                        verify_installation(current_python)
                else:
                    print("已取消")
            else:
                print("无效的选择")
        except ValueError:
            print("无效的输入")
    else:
        print("\n" + "=" * 80)
        print("手动复制方法：")
        print("=" * 80)
        print("\n如果你知道另一个环境的路径，可以手动复制：")
        print("\n1. 找到源环境的 site-packages 目录")
        print("   例如: C:\\Users\\YourName\\anaconda3\\envs\\your_env\\Lib\\site-packages")
        print("\n2. 复制以下文件夹到目标环境:")
        print(f"   {current_env}\\Lib\\site-packages\\")
        print("   - torch/")
        print("   - torch-*.dist-info/")
        print("   - torchvision/")
        print("   - torchvision-*.dist-info/")
        print("   - torchaudio/")
        print("   - torchaudio-*.dist-info/")
        print("   - nvidia/ (如果有)")
        print("   - nvfuser/ (如果有)")
        print("\n或者，提供源环境路径运行：")
        print("  python scripts/copy_torch_from_env.py --source <源环境路径>")


if __name__ == "__main__":
    main()
