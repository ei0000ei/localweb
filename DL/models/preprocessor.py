"""数据预处理类"""
import re
import jieba
import pandas as pd
from typing import Dict, List, Optional
import config.labels as labels

class TextPreprocessor:
    def __init__(self):
        # 初始化jieba，添加招聘领域词汇
        self._init_jieba()
        
        # 编译正则表达式模式
        self.patterns = self._compile_patterns()
    
    def _init_jieba(self):
        """初始化jieba分词器"""
        # 添加招聘领域特定词汇
        domain_words = [
            '五险一金', '周末双休', '带薪年假', '年终奖', '绩效奖金',
            '股票期权', '补充医疗保险', '定期体检', '餐补', '交通补助',
            '租房补贴', '通讯补贴', '团建活动', '弹性工作', '扁平管理'
        ]
        
        for word in domain_words:
            jieba.add_word(word)
    
    def _compile_patterns(self):
        """编译正则表达式模式"""
        return {
            'salary': [
                r'(\d+[\d~\-—–至]*\d*)\s*[万千wkK]?\s*[/\-~—–至]\s*(\d+[\d~\-—–至]*\d*)\s*[万千wkK]',
                r'薪资?[：:]\s*(\d+[\d~\-—–至]*\d*)\s*[万千wkK]?\s*[/\-~—–至]?\s*(\d+[\d~\-—–至]*\d*)\s*[万千wkK]?',
                r'(\d+)\s*[万千wkK]?\s*[/\-~—–至]\s*(\d+)\s*[万千wkK]',
            ],
            'experience': [
                r'(\d+[\d\-~]*\d*)\s*年经验',
                r'经验[：:]\s*(\d+[\d\-~]*\d*)\s*年',
                r'工作经历[：:]\s*(\d+[\d\-~]*\d*)\s*年',
                r'(\d+)\s*年及以上',
            ],
            'education': [
                r'(大专|本科|硕士|博士|中专|高中)及以上?',
                r'学历[：:]\s*(大专|本科|硕士|博士|中专|高中)',
                r'要求[：:].*?(大专|本科|硕士|博士|中专|高中)',
            ],
            'company_scale': [
                r'(\d+)\s*人以上',
                r'(\d+)\s*[-~至]\s*(\d+)\s*人',
                r'规模[：:]\s*(\d+)\s*人',
            ]
        }
    
    def clean_text(self, text: str) -> str:
        """清洗文本"""
        if not isinstance(text, str):
            return ""
        
        # 移除HTML标签
        text = re.sub(r'<[^>]+>', '', text)
        
        # 统一标点符号
        text = re.sub(r'[~—–]', '-', text)
        
        # 移除特殊字符但保留关键信息字符
        text = re.sub(r'[^\w\u4e00-\u9fff\-\d\.~万千wkK年月个以上以下]', ' ', text)
        
        # 合并多个空格
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def extract_salary(self, text: str) -> Optional[Dict]:
        """提取薪资信息"""
        for pattern in self.patterns['salary']:
            matches = re.findall(pattern, text)
            if matches:
                for match in matches:
                    if isinstance(match, tuple) and len(match) == 2:
                        min_sal, max_sal = match
                        try:
                            min_sal = self._convert_salary_unit(min_sal)
                            max_sal = self._convert_salary_unit(max_sal)
                            return {'min': float(min_sal), 'max': float(max_sal)}
                        except:
                            continue
        return None
    
    def _convert_salary_unit(self, salary_str: str) -> float:
        """转换薪资单位"""
        salary_str = str(salary_str).strip()
        
        # 处理"万"单位
        if '万' in salary_str:
            num = re.findall(r'\d+\.?\d*', salary_str)
            if num:
                return float(num[0]) * 10  # 转换为千元
        
        # 处理"k"单位
        if 'k' in salary_str.lower():
            num = re.findall(r'\d+\.?\d*', salary_str)
            if num:
                return float(num[0])
        
        # 默认认为是千元
        num = re.findall(r'\d+\.?\d*', salary_str)
        if num:
            return float(num[0])
        
        return 0.0
    
    def extract_experience(self, text: str) -> Optional[str]:
        """提取经验要求"""
        for pattern in self.patterns['experience']:
            matches = re.findall(pattern, text)
            if matches:
                exp = matches[0]
                if exp in ['1', '1-3', '1~3']:
                    return '1-3年'
                elif exp in ['3', '3-5', '3~5']:
                    return '3-5年'
                elif exp in ['5', '5-10', '5~10']:
                    return '5-10年'
                elif exp in ['10']:
                    return '10年以上'
                else:
                    return f"{exp}年"
        return None
    
    def extract_education(self, text: str) -> Optional[str]:
        """提取学历要求"""
        for pattern in self.patterns['education']:
            matches = re.findall(pattern, text)
            if matches:
                return matches[0]
        
        # 基于关键词匹配
        edu_keywords = {
            '博士': ['博士', '博士生'],
            '硕士': ['硕士', '研究生'],
            '本科': ['本科', '学士'],
            '大专': ['大专', '专科'],
            '中专': ['中专'],
            '高中': ['高中']
        }
        
        for edu_level, keywords in edu_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    return edu_level
        
        return None
    
    def preprocess_dataframe(self, df: pd.DataFrame, text_column: str = 'text') -> pd.DataFrame:
        """预处理整个DataFrame"""
        processed_df = df.copy()
        
        # 清洗文本
        processed_df['cleaned_text'] = processed_df[text_column].apply(self.clean_text)
        
        # 提取结构化信息
        processed_df['salary_info'] = processed_df['cleaned_text'].apply(self.extract_salary)
        processed_df['experience_info'] = processed_df['cleaned_text'].apply(self.extract_experience)
        processed_df['education_info'] = processed_df['cleaned_text'].apply(self.extract_education)
        
        return processed_df