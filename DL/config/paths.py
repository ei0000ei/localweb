"""路径配置文件"""
import os
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent

# 数据路径
DATA_PATHS = {
    'raw_data': PROJECT_ROOT / 'data' / 'raw',
    'processed_data': PROJECT_ROOT / 'data' / 'processed',
    'train_file': PROJECT_ROOT / 'data' / 'raw' / 'train.csv',
    'test_file': PROJECT_ROOT / 'data' / 'raw' / 'test.csv',
}

# 模型路径
MODEL_PATHS = {
    'bert_model': 'bert-base-chinese',
    'save_dir': PROJECT_ROOT / 'saved_models',
    'checkpoint_dir': PROJECT_ROOT / 'checkpoints',
}

# 输出路径
OUTPUT_PATHS = {
    'logs': PROJECT_ROOT / 'logs',
    'results': PROJECT_ROOT / 'results',
    'visualizations': PROJECT_ROOT / 'visualizations',
}

# 创建必要的目录
for path_group in [DATA_PATHS, MODEL_PATHS, OUTPUT_PATHS]:
    for key, path in path_group.items():
        if isinstance(path, Path):
            path.mkdir(parents=True, exist_ok=True)