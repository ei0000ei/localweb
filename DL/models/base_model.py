"""基础模型类"""
import torch
import torch.nn as nn
from transformers import BertModel, BertConfig

class BaseModel(nn.Module):
    def __init__(self, model_name='bert-base-chinese'):
        super(BaseModel, self).__init__()
        self.bert_config = BertConfig.from_pretrained(model_name)
        self.bert = BertModel.from_pretrained(model_name)
        
        # 冻结BERT的前几层
        self.freeze_bert_layers()
    
    def freeze_bert_layers(self, num_frozen_layers=6):
        """冻结BERT的前几层"""
        for i, (name, param) in enumerate(self.bert.named_parameters()):
            if i < num_frozen_layers:
                param.requires_grad = False
    
    def get_bert_embeddings(self, input_ids, attention_mask):
        """获取BERT嵌入"""
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        # 使用最后四层的平均作为文本表示
        hidden_states = outputs.hidden_states
        last_four_layers = hidden_states[-4:]
        stacked_hidden = torch.stack(last_four_layers, dim=0)
        mean_hidden = torch.mean(stacked_hidden, dim=0)
        
        return mean_hidden, outputs.pooler_output
    
    def forward(self, input_ids, attention_mask):
        raise NotImplementedError("子类必须实现forward方法")