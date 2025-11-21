"""
自动化脚本：将HAKE集成应用到FedE代码库

使用方法：
1. 设置FedE代码库路径
2. 运行脚本自动应用所有修改
"""

import os
import shutil
import sys
from pathlib import Path


def apply_integration(fede_path):
    """
    将HAKE集成应用到FedE代码库
    
    Args:
        fede_path: FedE代码库的路径
    """
    fede_path = Path(fede_path)
    integrated_path = Path(__file__).parent / 'fede_integrated'
    
    if not fede_path.exists():
        print(f"❌ 错误: FedE代码库路径不存在: {fede_path}")
        return False
    
    if not integrated_path.exists():
        print(f"❌ 错误: 集成代码目录不存在: {integrated_path}")
        return False
    
    print("=" * 60)
    print("FedE-HAKE集成自动应用脚本")
    print("=" * 60)
    print(f"\nFedE代码库路径: {fede_path}")
    print(f"集成代码路径: {integrated_path}\n")
    
    # 需要替换的文件列表
    files_to_replace = [
        'kge_model.py',
        'fede.py',
        'dataloader.py',
        'main.py',
        'kge_trainer.py',
    ]
    
    # 备份原始文件
    backup_dir = fede_path / 'backup_before_hake_integration'
    if not backup_dir.exists():
        backup_dir.mkdir()
        print(f"📦 创建备份目录: {backup_dir}")
    
    success_count = 0
    skip_count = 0
    
    for filename in files_to_replace:
        src_file = integrated_path / filename
        dst_file = fede_path / filename
        
        if not src_file.exists():
            print(f"⚠️  警告: 源文件不存在: {src_file}")
            continue
        
        if dst_file.exists():
            # 备份原文件
            backup_file = backup_dir / filename
            shutil.copy2(dst_file, backup_file)
            print(f"📋 已备份: {filename} -> backup_before_hake_integration/{filename}")
        
        # 复制新文件
        try:
            shutil.copy2(src_file, dst_file)
            print(f"✅ 已更新: {filename}")
            success_count += 1
        except Exception as e:
            print(f"❌ 更新失败: {filename} - {e}")
    
    # 检查fusion.py（可选文件）
    fusion_src = integrated_path / 'fusion.py'
    fusion_dst = fede_path / 'fusion.py'
    if fusion_src.exists():
        if fusion_dst.exists():
            backup_file = backup_dir / 'fusion.py'
            shutil.copy2(fusion_dst, backup_file)
        shutil.copy2(fusion_src, fusion_dst)
        print(f"✅ 已更新: fusion.py")
    
    print("\n" + "=" * 60)
    print(f"✅ 集成完成！")
    print(f"   - 成功更新: {success_count} 个文件")
    print(f"   - 备份位置: {backup_dir}")
    print("=" * 60)
    
    print("\n📝 下一步：")
    print("1. 检查修改是否正确")
    print("2. 运行测试验证")
    print("3. 开始训练HAKE模型")
    
    return True


def main():
    """主函数"""
    if len(sys.argv) > 1:
        fede_path = sys.argv[1]
    else:
        # 交互式输入
        print("FedE-HAKE集成自动应用脚本")
        print("=" * 60)
        fede_path = input("\n请输入FedE代码库的路径: ").strip()
        
        if not fede_path:
            print("❌ 错误: 未提供FedE代码库路径")
            print("\n使用方法:")
            print("  python apply_fede_integration.py <FedE代码库路径>")
            print("\n或者交互式运行:")
            print("  python apply_fede_integration.py")
            return 1
    
    success = apply_integration(fede_path)
    return 0 if success else 1


if __name__ == '__main__':
    exit(main())

