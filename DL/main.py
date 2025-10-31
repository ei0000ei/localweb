"""主程序入口"""
import argparse
from scripts.train import main as train_main
from scripts.predict import main as predict_main
from scripts.evaluate import main as evaluate_main

def main():
    parser = argparse.ArgumentParser(description='招聘信息提取系统')
    parser.add_argument('--mode', choices=['train', 'predict', 'evaluate'], 
                       required=True, help='运行模式')
    parser.add_argument('--data', help='数据文件路径')
    parser.add_argument('--model', help='模型文件路径')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train_main()
    elif args.mode == 'predict':
        predict_main()
    elif args.mode == 'evaluate':
        evaluate_main()

if __name__ == "__main__":
    main()