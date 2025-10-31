"""多任务模型"""
import torch
import torch.nn as nn
from .base_model import BaseModel
import config.model_config as model_config
import config.labels as labels

class MultiTaskJobModel(BaseModel):
    def __init__(self):
        super(MultiTaskJobModel, self).__init__(
            model_config.ModelConfig.BERT_CONFIG['model_name']
        )
        
        self.config = model_config.ModelConfig
        self.bert_dim = self.bert_config.hidden_size
        
        # 共享特征提取层
        self.shared_layers = self._build_shared_layers()
        
        # 多任务输出头
        self.task_heads = nn.ModuleDict()
        for task_name, task_config in self.config.TASK_HEADS.items():
            self.task_heads[task_name] = self._build_task_head(
                task_name, 
                task_config
            )
    
    def _build_shared_layers(self):
        """构建共享层"""
        return nn.Sequential(
            nn.Linear(self.bert_dim, self.config.MODEL_STRUCTURE['shared_hidden_size']),
            nn.ReLU(),
            nn.Dropout(self.config.MODEL_STRUCTURE['dropout_rate']),
            nn.Linear(self.config.MODEL_STRUCTURE['shared_hidden_size'], 
                     self.config.MODEL_STRUCTURE['task_hidden_size']),
            nn.ReLU(),
            nn.Dropout(self.config.MODEL_STRUCTURE['dropout_rate'] // 2)
        )
    
    def _build_task_head(self, task_name, task_config):
        """构建任务特定的输出头"""
        hidden_size = self.config.MODEL_STRUCTURE['task_hidden_size']
        num_classes = task_config['num_classes']
        
        if 'salary' in task_name and 'class' not in task_name:
            # 薪资回归任务
            return nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_size // 2, 1)
            )
        else:
            # 分类任务
            return nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_size // 2, num_classes)
            )
    
    def forward(self, input_ids, attention_mask):
        # 获取BERT特征
        sequence_output, pooled_output = self.get_bert_embeddings(
            input_ids, attention_mask
        )
        
        # 使用[CLS] token的特征
        cls_features = pooled_output
        
        # 共享特征提取
        shared_features = self.shared_layers(cls_features)
        
        # 多任务预测
        outputs = {}
        for task_name, task_head in self.task_heads.items():
            task_output = task_head(shared_features)
            
            if 'salary' in task_name and 'class' not in task_name:
                # 对薪资输出应用ReLU确保非负
                task_output = torch.relu(task_output)
            
            outputs[task_name] = task_output
        
        return outputs