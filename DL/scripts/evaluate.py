"""评估脚本"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import torch
from inference.predictor import JobInfoPredictor
from training.metrics import MultiTaskMetrics
from utils.helpers import get_device
import config.model_config as model_config
import config.paths as paths

def main():
    # 初始化预测器和设备
    device = get_device()
    predictor = JobInfoPredictor(
        model_path=paths.MODEL_PATHS['save_dir'] / 'best_model.pt',
        device=device
    )
    
    # 加载测试数据
    test_df = pd.read_csv(paths.DATA_PATHS['test_file'])
    print(f"测试集大小: {len(test_df)}")
    
    # 进行预测
    predictions = predictor.predict_dataframe(test_df)
    
    # 计算评估指标
    metrics_calculator = MultiTaskMetrics(model_config.ModelConfig.TASK_HEADS)
    
    # 这里需要将预测结果转换为模型输出格式进行比较
    # 实际应用中需要根据具体的数据格式进行调整
    
    # 保存评估结果
    evaluation_results = {
        'total_samples': len(test_df),
        'predictions': predictions.to_dict('records')
    }
    
    results_path = paths.OUTPUT_PATHS['results'] / 'evaluation_results.json'
    
    import json
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(evaluation_results, f, ensure_ascii=False, indent=2)
    
    print(f"评估结果已保存至: {results_path}")

if __name__ == "__main__":
    main()