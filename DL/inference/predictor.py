"""预测器"""
import torch
import pandas as pd
from typing import List, Dict, Any
from models.multi_task_model import MultiTaskJobModel
from models.preprocessor import TextPreprocessor
import config.model_config as model_config
import config.labels as labels

class JobInfoPredictor:
    def __init__(self, model_path: str, device: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model(model_path)
        self.preprocessor = TextPreprocessor()
        self.config = model_config.ModelConfig
        
        # 标签映射
        self.label_maps = {
            'position': labels.POSITION_LABELS,
            'education': labels.EDUCATION_LABELS,
            'experience': labels.EXPERIENCE_LABELS,
            'company_type': labels.COMPANY_TYPE_LABELS,
            'scale': labels.SCALE_LABELS,
            'salary_class': labels.SALARY_CLASS_LABELS
        }
    
    def _load_model(self, model_path: str) -> MultiTaskJobModel:
        """加载模型"""
        model = MultiTaskJobModel()
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        return model
    
    def predict_single(self, text: str) -> Dict[str, Any]:
        """预测单条文本"""
        # 预处理文本
        cleaned_text = self.preprocessor.clean_text(text)
        
        # 规则提取（作为后备）
        rule_based_results = self._rule_based_extraction(text)
        
        # 模型预测
        model_results = self._model_prediction(cleaned_text)
        
        # 融合结果
        final_results = self._merge_results(model_results, rule_based_results, text)
        
        return final_results
    
    def _model_prediction(self, text: str) -> Dict[str, Any]:
        """模型预测"""
        from transformers import BertTokenizer
        
        tokenizer = BertTokenizer.from_pretrained(self.config.BERT_CONFIG['model_name'])
        
        # Tokenize
        inputs = tokenizer(
            text,
            max_length=self.config.BERT_CONFIG['max_length'],
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
        
        # 解析输出
        results = {}
        
        for task_name, output in outputs.items():
            if 'salary' in task_name and 'class' not in task_name:
                # 回归任务
                value = output.squeeze().item()
                if task_name == 'salary_min':
                    results['salary_min'] = round(value, 1)
                else:
                    results['salary_max'] = round(value, 1)
            else:
                # 分类任务
                pred_class = torch.argmax(output, dim=1).item()
                label_map = self.label_maps.get(task_name, [])
                if label_map and pred_class < len(label_map):
                    results[task_name] = label_map[pred_class]
                else:
                    results[task_name] = pred_class
        
        return results
    
    def _rule_based_extraction(self, text: str) -> Dict[str, Any]:
        """基于规则的信息提取"""
        results = {}
        
        # 提取薪资
        salary_info = self.preprocessor.extract_salary(text)
        if salary_info:
            results['salary_min'] = salary_info['min']
            results['salary_max'] = salary_info['max']
        
        # 提取经验
        experience = self.preprocessor.extract_experience(text)
        if experience:
            results['experience'] = experience
        
        # 提取学历
        education = self.preprocessor.extract_education(text)
        if education:
            results['education'] = education
        
        return results
    
    def _merge_results(self, model_results: Dict, rule_results: Dict, original_text: str) -> Dict[str, Any]:
        """融合模型结果和规则结果"""
        final_results = {
            'original_text': original_text,
            'extracted_info': {}
        }
        
        # 优先使用规则提取的数值信息
        for key in ['salary_min', 'salary_max']:
            if key in rule_results:
                final_results['extracted_info'][key] = rule_results[key]
            elif key in model_results:
                final_results['extracted_info'][key] = model_results[key]
        
        # 分类信息使用模型结果
        for key in ['position', 'education', 'experience', 'company_type', 'scale']:
            if key in model_results:
                final_results['extracted_info'][key] = model_results[key]
        
        # 生成薪资范围字符串
        if 'salary_min' in final_results['extracted_info'] and 'salary_max' in final_results['extracted_info']:
            min_sal = final_results['extracted_info']['salary_min']
            max_sal = final_results['extracted_info']['salary_max']
            final_results['extracted_info']['salary_range'] = f"{min_sal}-{max_sal}k"
        
        return final_results
    
    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """批量预测"""
        results = []
        for text in texts:
            result = self.predict_single(text)
            results.append(result)
        return results
    
    def predict_dataframe(self, df: pd.DataFrame, text_column: str = 'text') -> pd.DataFrame:
        """预测DataFrame"""
        texts = df[text_column].tolist()
        predictions = self.predict_batch(texts)
        
        # 将预测结果转换为DataFrame
        extracted_info_list = [pred['extracted_info'] for pred in predictions]
        result_df = pd.DataFrame(extracted_info_list)
        
        return pd.concat([df, result_df], axis=1)