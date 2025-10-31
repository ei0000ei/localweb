"""训练脚本"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
from data.dataloader import DataLoaderManager
from models.multi_task_model import MultiTaskJobModel
from training.trainer import Trainer
from utils.helpers import set_seed, get_device
import config.model_config as model_config
import config.paths as paths

def main():
    # 设置随机种子
    set_seed(42)
    
    # 获取设备
    device = get_device()
    print(f"使用设备: {device}")
    
    # 加载数据
    data_loader_manager = DataLoaderManager(paths.DATA_PATHS['train_file'])
    df = data_loader_manager.load_data()
    
    if df is None:
        print("数据加载失败，请检查数据文件路径")
        return
    
    # 创建数据加载器
    train_loader, val_loader = data_loader_manager.create_data_loaders(
        df, 
        batch_size=model_config.ModelConfig.TRAINING['batch_size']
    )
    
    # 初始化模型
    model = MultiTaskJobModel()
    model.to(device)
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # 初始化训练器
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        config=model_config.ModelConfig
    )
    
    # 开始训练
    trainer.train()
    
    print("训练完成!")

if __name__ == "__main__":
    main()