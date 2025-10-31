"""后处理器"""
import re
from typing import Dict, Any

class ResultPostProcessor:
    def __init__(self):
        self.validation_rules = self._setup_validation_rules()
    
    def _setup_validation_rules(self):
        """设置验证规则"""
        return {
            'salary': {
                'min': 0,
                'max': 100,  # 假设最高100k
                'logic': lambda x: x['min'] <= x['max']
            },
            'experience': {
                'allowed_values': ['无经验', '1年以下', '1-3年', '3-5年', '5-10年', '10年以上']
            },
            'education': {
                'allowed_values': ['不限', '中专', '高中', '大专', '本科', '硕士', '博士']
            }
        }
    
    def validate_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """验证结果合理性"""
        validated_results = results.copy()
        
        # 验证薪资范围
        if 'salary_min' in validated_results and 'salary_max' in validated_results:
            min_sal = validated_results['salary_min']
            max_sal = validated_results['salary_max']
            
            # 检查数值范围
            if min_sal < self.validation_rules['salary']['min']:
                validated_results['salary_min'] = self.validation_rules['salary']['min']
            if max_sal > self.validation_rules['salary']['max']:
                validated_results['salary_max'] = self.validation_rules['salary']['max']
            
            # 检查逻辑合理性
            if min_sal > max_sal:
                # 交换最小值最大值
                validated_results['salary_min'], validated_results['salary_max'] = \
                    validated_results['salary_max'], validated_results['salary_min']
        
        # 验证经验要求
        if 'experience' in validated_results:
            exp = validated_results['experience']
            if exp not in self.validation_rules['experience']['allowed_values']:
                validated_results['experience'] = '经验不详'
        
        # 验证学历要求
        if 'education' in validated_results:
            edu = validated_results['education']
            if edu not in self.validation_rules['education']['allowed_values']:
                validated_results['education'] = '学历不限'
        
        return validated_results
    
    def format_output(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """格式化输出"""
        formatted = {}
        
        # 薪资格式化
        if 'salary_min' in results and 'salary_max' in results:
            formatted['薪资范围'] = f"{results['salary_min']}-{results['salary_max']}k/月"
        
        # 其他字段格式化
        field_mapping = {
            'position': '职位',
            'experience': '经验要求',
            'education': '学历要求',
            'company_type': '企业性质',
            'scale': '企业规模'
        }
        
        for eng_key, chn_key in field_mapping.items():
            if eng_key in results:
                formatted[chn_key] = results[eng_key]
        
        return formatted