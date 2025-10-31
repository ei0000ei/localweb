"""预测脚本"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
from inference.predictor import JobInfoPredictor
import config.paths as paths

def main():
    # 初始化预测器
    predictor = JobInfoPredictor(
        model_path=paths.MODEL_PATHS['save_dir'] / 'best_model.pt'
    )
    
    # 示例文本
    sample_texts = [
        "招聘Java高级开发工程师，薪资25-40K，要求5年以上经验，本科学历，上市公司，规模1000人以上",
        "急聘前端开发工程师，薪资15-25k，3年经验，大专学历，民营企业，规模500人",
        "招聘产品经理，薪资30-50万/年，8年经验，硕士学历，外资企业"
    ]
    
    print("开始预测...")
    
    # 批量预测
    results = predictor.predict_batch(sample_texts)
    
    # 打印结果
    for i, (text, result) in enumerate(zip(sample_texts, results)):
        print(f"\n示例 {i+1}:")
        print(f"原始文本: {text}")
        print("提取结果:")
        for key, value in result['extracted_info'].items():
            print(f"  {key}: {value}")
    
    # 保存结果
    output_df = pd.DataFrame([
        {**result['extracted_info'], 'original_text': result['original_text']}
        for result in results
    ])
    
    output_path = paths.OUTPUT_PATHS['results'] / 'predictions.csv'
    output_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n预测结果已保存至: {output_path}")

if __name__ == "__main__":
    main()