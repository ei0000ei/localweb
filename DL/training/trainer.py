"""训练器"""
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import numpy as np
from tqdm import tqdm
import os

from .loss_functions import MultiTaskLoss
from .metrics import MultiTaskMetrics
from utils.logger import TrainingLogger

class Trainer:
    def __init__(self, model, train_loader, val_loader, device, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        
        # 训练组件
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        self.criterion = MultiTaskLoss(config.TASK_HEADS)
        self.metrics = MultiTaskMetrics(config.TASK_HEADS)
        self.logger = TrainingLogger()
        
        # 训练状态
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
    
    def _setup_optimizer(self):
        """设置优化器"""
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                'weight_decay': self.config.TRAINING['weight_decay'],
            },
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
            },
        ]
        
        return AdamW(
            optimizer_grouped_parameters,
            lr=self.config.TRAINING['learning_rate'],
            eps=1e-8
        )
    
    def _setup_scheduler(self):
        """设置学习率调度器"""
        num_training_steps = len(self.train_loader) * self.config.TRAINING['epochs']
        num_warmup_steps = int(num_training_steps * self.config.TRAINING['warmup_ratio'])
        
        return get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
    
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch+1} Training')
        
        for batch_idx, batch in enumerate(progress_bar):
            # 移动数据到设备
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(input_ids, attention_mask)
            
            # 计算损失
            loss_dict = self.criterion.compute_loss(outputs, batch, self.device)
            total_loss += loss_dict['total_loss'].item()
            
            # 反向传播
            loss_dict['total_loss'].backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.config.TRAINING['max_grad_norm']
            )
            
            # 更新参数
            self.optimizer.step()
            self.scheduler.step()
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': f'{loss_dict["total_loss"].item():.4f}',
                'lr': f'{self.scheduler.get_last_lr()[0]:.2e}'
            })
        
        avg_loss = total_loss / len(self.train_loader)
        self.train_losses.append(avg_loss)
        
        return avg_loss
    
    def validate_epoch(self, epoch):
        """验证一个epoch"""
        self.model.eval()
        total_loss = 0
        all_metrics = {}
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc=f'Epoch {epoch+1} Validation'):
                # 移动数据到设备
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                # 前向传播
                outputs = self.model(input_ids, attention_mask)
                
                # 计算损失
                loss_dict = self.criterion.compute_loss(outputs, batch, self.device)
                total_loss += loss_dict['total_loss'].item()
                
                # 计算指标
                batch_metrics = self.metrics.compute_metrics(outputs, batch, self.device)
                for task, metrics in batch_metrics.items():
                    if task not in all_metrics:
                        all_metrics[task] = {k: 0 for k in metrics.keys()}
                    for metric_name, value in metrics.items():
                        all_metrics[task][metric_name] += value
        
        # 平均指标
        avg_loss = total_loss / len(self.val_loader)
        self.val_losses.append(avg_loss)
        
        for task in all_metrics:
            for metric_name in all_metrics[task]:
                all_metrics[task][metric_name] /= len(self.val_loader)
        
        return avg_loss, all_metrics
    
    def save_checkpoint(self, epoch, is_best=False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss
        }
        
        # 保存最新检查点
        checkpoint_path = os.path.join(
            self.config.MODEL_PATHS['checkpoint_dir'], 
            f'checkpoint_epoch_{epoch}.pt'
        )
        torch.save(checkpoint, checkpoint_path)
        
        # 保存最佳模型
        if is_best:
            best_model_path = os.path.join(
                self.config.MODEL_PATHS['save_dir'],
                'best_model.pt'
            )
            torch.save(self.model.state_dict(), best_model_path)
    
    def train(self, epochs=None):
        """完整的训练流程"""
        if epochs is None:
            epochs = self.config.TRAINING['epochs']
        
        print("开始训练...")
        for epoch in range(epochs):
            # 训练阶段
            train_loss = self.train_epoch(epoch)
            
            # 验证阶段
            val_loss, val_metrics = self.validate_epoch(epoch)
            
            # 记录日志
            self.logger.log_epoch(epoch, train_loss, val_loss, val_metrics)
            
            # 保存检查点
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            
            self.save_checkpoint(epoch, is_best)
            
            # 打印进度
            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'  训练损失: {train_loss:.4f}')
            print(f'  验证损失: {val_loss:.4f}')
            print(f'  最佳验证损失: {self.best_val_loss:.4f}')
            
            # 打印任务指标
            for task, metrics in val_metrics.items():
                print(f'  {task}: {metrics}')
        
        print("训练完成!")
        return self.train_losses, self.val_losses