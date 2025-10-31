"""日志工具"""
import logging
import os
from datetime import datetime
import config.paths as paths

class TrainingLogger:
    def __init__(self, log_dir=None):
        self.log_dir = log_dir or paths.OUTPUT_PATHS['logs']
        self.setup_logging()
    
    def setup_logging(self):
        """设置日志"""
        log_filename = datetime.now().strftime("training_%Y%m%d_%H%M%S.log")
        log_path = os.path.join(self.log_dir, log_filename)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_path),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
    
    def log_epoch(self, epoch, train_loss, val_loss, metrics):
        """记录epoch信息"""
        self.logger.info(f"Epoch {epoch + 1}:")
        self.logger.info(f"  训练损失: {train_loss:.4f}")
        self.logger.info(f"  验证损失: {val_loss:.4f}")
        
        for task, task_metrics in metrics.items():
            self.logger.info(f"  {task}: {task_metrics}")
    
    def log_message(self, message, level='info'):
        """记录普通消息"""
        if level == 'info':
            self.logger.info(message)
        elif level == 'warning':
            self.logger.warning(message)
        elif level == 'error':
            self.logger.error(message)

class PredictionLogger:
    def __init__(self, log_dir=None):
        self.log_dir = log_dir or paths.OUTPUT_PATHS['logs']
        self.setup_logging()
    
    def setup_logging(self):
        """设置预测日志"""
        log_filename = datetime.now().strftime("prediction_%Y%m%d_%H%M%S.log")
        log_path = os.path.join(self.log_dir, log_filename)
        
        self.logger = logging.getLogger('prediction')
        self.logger.setLevel(logging.INFO)
        
        if not self.logger.handlers:
            handler = logging.FileHandler(log_path)
            formatter = logging.Formatter('%(asctime)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def log_prediction(self, original_text, prediction):
        """记录预测结果"""
        self.logger.info("原始文本: %s", original_text)
        self.logger.info("预测结果: %s", prediction)
        self.logger.info("-" * 50)