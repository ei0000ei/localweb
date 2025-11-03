"""模型配置参数"""
class ModelConfig:
    # BERT配置
    BERT_CONFIG = {
        # 'model_name': 'bert-base-chinese', # huggingface预训练模型名称，需下载
        'model_name': '/root/YP/DL_uniform/DL/bert-base-chinese-local', # 本地预训练模型路径
        'max_length': 256,
        'hidden_dropout_prob': 0.3,
        'attention_probs_dropout_prob': 0.3
    }
    
    # 模型结构配置
    MODEL_STRUCTURE = {
        'shared_hidden_size': 512,
        'task_hidden_size': 256,
        'dropout_rate': 0.3
    }
    
    # 多任务头配置
    TASK_HEADS = {
        'position': {'num_classes': 50, 'loss_weight': 1.0},
        'salary_min': {'num_classes': 1, 'loss_weight': 0.8, 'type': 'regression'},
        'salary_max': {'num_classes': 1, 'loss_weight': 0.8, 'type': 'regression'},
        'salary_class': {'num_classes': 6, 'loss_weight': 1.0},
        'experience': {'num_classes': 5, 'loss_weight': 1.0},
        'education': {'num_classes': 6, 'loss_weight': 1.0},
        'company_type': {'num_classes': 10, 'loss_weight': 1.0},
        'scale': {'num_classes': 8, 'loss_weight': 1.0},
        'company_attr': {'num_classes': 10, 'loss_weight': 1.0}
    }
    
    # 训练配置
    TRAINING = {
        'batch_size': 16,
        'learning_rate': 2e-5,
        'epochs': 20,
        'warmup_ratio': 0.1,
        'weight_decay': 0.01,
        'max_grad_norm': 1.0
    }
    
    # 数据配置
    DATA = {
        'test_size': 0.2,
        'random_state': 42,
        'shuffle': True
    }