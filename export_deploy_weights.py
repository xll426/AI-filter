#!/usr/bin/env python3
"""
导出部署模型权重脚本

该脚本从训练好的 MBRPUCNet 模型中导出可用于部署的权重文件。
部署权重文件经过结构重组和精度转换，可直接用于 iccv_yan_2025_deploy_model.py。

用法示例:
  python3 export_deploy_weights.py \
    --cfg options/inference/infer_iccv_yan_2025.yml \
    --output weights/iccv_yan_2025_deploy_fp16.pth
"""
import os
import argparse
import yaml
import torch
from pyjnd.api_helpers import get_model


def export_deploy_weights(config_path, output_path, device='cpu'):
    """
    从配置文件创建模型并导出部署权重
    
    Args:
        config_path (str): 模型配置文件路径 (YAML格式)
        output_path (str): 输出权重文件路径
        device (str): 运行设备 ('cpu' 或 'cuda')
    """
    # 1. 读取配置文件
    print(f"[INFO] 正在读取配置文件: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 2. 设置设备
    if device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda:0')
        print(f"[INFO] 使用 GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print(f"[INFO] 使用 CPU")
    
    # 3. 创建模型并加载权重
    print(f"[INFO] 正在创建模型: {config.get('type', 'Unknown')}")
    model = get_model(config, device)
    
    # 确认模型类型
    if not hasattr(model.net, 'export_deploy_weights'):
        raise AttributeError(
            f"模型 {type(model.net).__name__} 不支持 export_deploy_weights 方法。\n"
            "请确保使用的是 MBRPUCNet 模型。"
        )
    
    # 4. 导出部署权重
    print(f"[INFO] 正在导出部署权重...")
    model.eval()  # 切换到评估模式
    
    # 创建输出目录（如果不存在）
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    # 调用导出方法
    model.net.export_deploy_weights(output_path)
    
    # 5. 验证导出的权重
    print(f"[INFO] 正在验证导出的权重文件...")
    try:
        deploy_state_dict = torch.load(output_path, map_location='cpu')
        print(f"[SUCCESS] 导出成功！权重文件包含 {len(deploy_state_dict)} 个参数:")
        for key in deploy_state_dict.keys():
            shape = deploy_state_dict[key].shape
            dtype = deploy_state_dict[key].dtype
            print(f"  - {key}: {shape} ({dtype})")
    except Exception as e:
        print(f"[ERROR] 验证失败: {e}")
        return False
    
    print(f"\n[DONE] 部署权重已保存至: {output_path}")
    print(f"[INFO] 可使用以下代码加载部署模型:")
    print(f"```python")
    print(f"from iccv_yan_2025_deploy_model import PUCNet")
    print(f"import torch")
    print(f"")
    print(f"model = PUCNet(in_channels_expanded=16)")
    print(f"state_dict = torch.load('{output_path}', map_location='cpu')")
    print(f"model.load_state_dict(state_dict)")
    print(f"model.eval()")
    print(f"```")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="导出 MBRPUCNet 模型的部署权重",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--cfg", 
        required=True,
        help="模型配置文件路径 (YAML格式)"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="输出权重文件路径"
    )
    parser.add_argument(
        "--device",
        choices=['cpu', 'cuda'],
        default='cpu',
        help="运行设备 (默认: cpu)"
    )
    
    args = parser.parse_args()
    
    # 检查配置文件是否存在
    if not os.path.isfile(args.cfg):
        print(f"[ERROR] 配置文件不存在: {args.cfg}")
        return
    
    # 执行导出
    try:
        success = export_deploy_weights(args.cfg, args.output, args.device)
        if success:
            print(f"\n✓ 导出流程完成")
        else:
            print(f"\n✗ 导出流程失败")
    except Exception as e:
        print(f"[ERROR] 导出过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
