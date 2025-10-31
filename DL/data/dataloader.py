"""数据加载器"""
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import config.paths as paths
import config.model_config as model_config

class JobInfoDataset(Dataset):
    def __init__(self, texts, labels=None, tokenizer=None, is_training=True):
        self.texts = texts
        self.labels = labels
        self.is_training = is_training
        self.tokenizer = tokenizer or BertTokenizer.from_pretrained(
            model_config.ModelConfig.BERT_CONFIG['model_name']
        )
        self.max_length = model_config.ModelConfig.BERT_CONFIG['max_length']
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        result = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'text': text
        }
        
        # Add labels if training
        if self.is_training and self.labels is not None:
            label_row = self.labels.iloc[idx] if hasattr(self.labels, 'iloc') else self.labels[idx]
            
            # Convert labels to tensor format
            for task in model_config.ModelConfig.TASK_HEADS.keys():
                if task in label_row:
                    if 'salary' in task and 'class' not in task:
                        # Regression task for salary
                        result[task] = torch.tensor(float(label_row[task]), dtype=torch.float)
                    else:
                        # Classification task
                        result[task] = torch.tensor(int(label_row[task]), dtype=torch.long)
        
        return result

class DataLoaderManager:
    def __init__(self, data_path):
        self.data_path = data_path
        self.tokenizer = BertTokenizer.from_pretrained(
            model_config.ModelConfig.BERT_CONFIG['model_name']
        )
    
    def load_data(self):
        """加载数据"""
        try:
            df = pd.read_csv(self.data_path)
            print(f"成功加载数据，共 {len(df)} 条样本")
            return df
        except Exception as e:
            print(f"数据加载失败: {e}")
            return None
    
    def create_data_loaders(self, df, batch_size=16):
        """创建训练和验证数据加载器"""
        from sklearn.model_selection import train_test_split
        
        # 分割数据
        train_df, val_df = train_test_split(
            df, 
            test_size=model_config.ModelConfig.DATA['test_size'],
            random_state=model_config.ModelConfig.DATA['random_state'],
            shuffle=model_config.ModelConfig.DATA['shuffle']
        )
        
        # 创建数据集
        train_dataset = JobInfoDataset(
            texts=train_df['text'].values,
            labels=train_df,
            tokenizer=self.tokenizer,
            is_training=True
        )
        
        val_dataset = JobInfoDataset(
            texts=val_df['text'].values,
            labels=val_df,
            tokenizer=self.tokenizer,
            is_training=True
        )
        
        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2
        )
        
        print(f"训练集: {len(train_dataset)} 条样本")
        print(f"验证集: {len(val_dataset)} 条样本")
        
        return train_loader, val_loader
    
    def create_prediction_loader(self, texts, batch_size=16):
        """创建预测数据加载器"""
        dataset = JobInfoDataset(
            texts=texts,
            labels=None,
            tokenizer=self.tokenizer,
            is_training=False
        )
        
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2
        )
        
        return loader