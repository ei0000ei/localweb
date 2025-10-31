"""评估指标"""
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error

class MultiTaskMetrics:
    def __init__(self, task_configs):
        self.task_configs = task_configs
    
    def compute_metrics(self, outputs, batch, device):
        """计算多任务指标"""
        metrics = {}
        
        for task_name in self.task_configs.keys():
            if task_name not in batch:
                continue
                
            pred = outputs[task_name]
            target = batch[task_name].to(device)
            
            if 'salary' in task_name and 'class' not in task_name:
                # 回归任务指标
                pred = pred.squeeze().detach().cpu().numpy()
                target = target.float().detach().cpu().numpy()
                
                mse = mean_squared_error(target, pred)
                mae = np.mean(np.abs(target - pred))
                
                metrics[task_name] = {
                    'mse': mse,
                    'mae': mae
                }
            else:
                # 分类任务指标
                pred_class = torch.argmax(pred, dim=1).detach().cpu().numpy()
                target_class = target.detach().cpu().numpy()
                
                accuracy = accuracy_score(target_class, pred_class)
                f1 = f1_score(target_class, pred_class, average='weighted')
                
                metrics[task_name] = {
                    'accuracy': accuracy,
                    'f1_score': f1
                }
        
        return metrics