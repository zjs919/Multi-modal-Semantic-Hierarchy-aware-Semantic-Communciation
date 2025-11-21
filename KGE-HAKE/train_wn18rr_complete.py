#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
WN18RR数据集完整训练脚本
支持训练、验证和测试
"""

import os
import sys
import subprocess

def train_hake_wn18rr(gpu_id=0, save_id=0):
    """训练HAKE模型在WN18RR数据集上"""
    
    # 设置GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    # 训练命令
    cmd = [
        sys.executable, '-u', 'codes/runs.py',
        '--do_train',
        '--do_valid',
        '--do_test',
        '--data_path', 'data/wn18rr',
        '--model', 'HAKE',
        '-n', '1024',              # negative_sample_size
        '-b', '512',               # batch_size
        '-d', '500',               # hidden_dim
        '-g', '6.0',               # gamma
        '-a', '0.5',               # adversarial_temperature
        '-lr', '0.00005',          # learning_rate
        '--max_steps', '80000',    # max_steps
        '-save', f'models/HAKE_wn18rr_{save_id}',
        '--test_batch_size', '8',
        '-mw', '0.5',              # modulus_weight
        '-pw', '0.5'               # phase_weight
    ]
    
    print("=" * 60)
    print("开始训练HAKE模型在WN18RR数据集上")
    print("=" * 60)
    print(f"GPU设备: {gpu_id}")
    print(f"保存路径: models/HAKE_wn18rr_{save_id}")
    print("=" * 60)
    
    # 运行训练
    subprocess.run(cmd)


def test_model(model_path, gpu_id=0):
    """测试已训练的模型"""
    
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    cmd = [
        sys.executable, '-u', 'codes/runs.py',
        '--do_test',
        '-init', model_path
    ]
    
    print("=" * 60)
    print(f"测试模型: {model_path}")
    print("=" * 60)
    
    subprocess.run(cmd)


def valid_model(model_path, gpu_id=0):
    """在验证集上评估模型"""
    
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    cmd = [
        sys.executable, '-u', 'codes/runs.py',
        '--do_valid',
        '-init', model_path
    ]
    
    print("=" * 60)
    print(f"验证模型: {model_path}")
    print("=" * 60)
    
    subprocess.run(cmd)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='训练/测试HAKE模型在WN18RR上')
    parser.add_argument('--mode', type=str, choices=['train', 'test', 'valid'],
                       default='train', help='运行模式')
    parser.add_argument('--gpu', type=int, default=0, help='GPU设备ID')
    parser.add_argument('--save_id', type=int, default=0, help='保存ID')
    parser.add_argument('--model_path', type=str, default=None,
                       help='模型路径（用于test/valid模式）')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train_hake_wn18rr(args.gpu, args.save_id)
    elif args.mode == 'test':
        if args.model_path is None:
            args.model_path = f'models/HAKE_wn18rr_{args.save_id}'
        test_model(args.model_path, args.gpu)
    elif args.mode == 'valid':
        if args.model_path is None:
            args.model_path = f'models/HAKE_wn18rr_{args.save_id}'
        valid_model(args.model_path, args.gpu)

