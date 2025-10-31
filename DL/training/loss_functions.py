"""损失函数"""
import torch
import torch.nn as nn

class MultiTaskLoss:
    def __init__(self, task_configs):
        self.task_configs = task_configs
        self.regression_loss = nn.MSELoss()
        self.classification_loss = nn.CrossEntropyLoss()
    
    def compute_loss(self, outputs, batch, device):
        """计算多任务损失"""
        total_loss = 0
        loss_dict = {}
        
        for task_name, task_config in self.task_configs.items():
            if task_name not in batch:
                continue
                
            weight = task_config['loss_weight']
            pred = outputs[task_name]
            target = batch[task_name].to(device)
            
            if 'salary' in task_name and 'class' not in task_name:
                # 回归任务
                pred = pred.squeeze()
                if len(pred.shape) == 0:
                    pred = pred.unsqueeze(0)
                loss = self.regression_loss(pred, target.float())
            else:
                # 分类任务
                loss = self.classification_loss(pred, target)
            
            weighted_loss = weight * loss
            total_loss += weighted_loss
            loss_dict[task_name] = loss.item()
        
        loss_dict['total_loss'] = total_loss
        return loss_dict