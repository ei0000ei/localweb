"""辅助函数"""
import torch
import numpy as np
import random
import json
from typing import Any, Dict

def set_seed(seed: int = 42):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def save_json(data: Dict[str, Any], filepath: str):
    """保存JSON文件"""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_json(filepath: str) -> Dict[str, Any]:
    """加载JSON文件"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def count_parameters(model):
    """计算模型参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_device():
    """获取设备"""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def format_salary(salary_min, salary_max, unit='k'):
    """格式化薪资显示"""
    if unit == 'k':
        return f"{salary_min}-{salary_max}k/月"
    elif unit == '万':
        return f"{salary_min/10:.1f}-{salary_max/10:.1f}万/月"