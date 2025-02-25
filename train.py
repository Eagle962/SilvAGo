#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
圍棋AI主訓練腳本
"""

import os
import sys
import argparse
import json
import logging
import tensorflow as tf
import numpy as np
from datetime import datetime

# 確保可以導入自定義模組
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 導入自定義模組
from game.go_game import GoGame
from ml.model import create_go_model, create_light_model, create_cnn_rnn_model
from ml.training import TrainingPipeline
from data.dataset import process_sgf_files, load_data
from search.mcts import MCTS

# 設置日誌記錄
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('train.log')
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """解析命令行參數。"""
    parser = argparse.ArgumentParser(description='圍棋AI訓練腳本')
    
    # 訓練模式
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'selfplay', 'process_data', 'eval'],
                       help='訓練模式: train-完整訓練, selfplay-僅自我對弈, process_data-處理sgf數據, eval-評估模型')
    
    # 數據相關
    parser.add_argument('--sgf_dir', type=str, default='data/sgf',
                       help='SGF文件目錄')
    parser.add_argument('--data_file', type=str, default='data/processed_data.npz',
                       help='處理後的數據文件')
    parser.add_argument('--min_rank', type=int, default=7,
                       help='最低段位過濾(默認7段以上)')
    
    # 模型相關
    parser.add_argument('--model_type', type=str, default='residual', choices=['residual', 'light', 'cnn_rnn'],
                       help='模型類型')
    parser.add_argument('--model_path', type=str, default=None,
                       help='預訓練模型路徑(如果存在)')
    parser.add_argument('--output_dir', type=str, default='models',
                       help='模型輸出目錄')
    
    # 訓練參數
    parser.add_argument('--batch_size', type=int, default=256,
                       help='批次大小')
    parser.add_argument('--epochs', type=int, default=10,
                       help='每次迭代的訓練輪數')
    parser.add_argument('--iterations', type=int, default=20,
                       help='訓練迭代次數')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='初始學習率')
    parser.add_argument('--use_mixed_precision', action='store_true',
                       help='使用混合精度訓練')
    
    # 自我對弈參數
    parser.add_argument('--num_games', type=int, default=500,
                       help='自我對弈遊戲數量')
    parser.add_argument('--mcts_sims', type=int, default=800,
                       help='MCTS模擬次數')
    
    # 系統相關
    parser.add_argument('--gpu', type=str, default=None,
                       help='指定GPU設備(例如 "0,1")')
    parser.add_argument('--verbose', action='store_true',
                       help='詳細輸出')
    
    return parser.parse_args()

def set_gpu_options(gpu_str=None):
    """設置GPU選項。"""
    if gpu_str is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_str
        logger.info(f"使用GPU設備: {gpu_str}")
    
    # 設置GPU成長模式
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"發現 {len(gpus)} 個GPU設備，已設置記憶體增長模式")
        except RuntimeError as e:
            logger.error(f"設置GPU記憶體增長模式失敗: {str(e)}")

def create_model(args):
    """創建模型。"""
    # 定義輸入形狀
    input_shape = (19, 19, 17)  # 標準圍棋特徵平面
    
    # 根據模型類型創建模型
    if args.model_type == 'residual':
        model = create_go_model(input_shape=input_shape, num_filters=256, num_res_blocks=19)
        logger.info("創建完整殘差網絡模型")
    elif args.model_type == 'light':
        model = create_light_model(input_shape=input_shape, num_filters=128, num_res_blocks=10)
        logger.info("創建輕量級殘差網絡模型")
    elif args.model_type == 'cnn_rnn':
        model = create_cnn_rnn_model(board_input_shape=input_shape)
        logger.info("創建CNN-RNN混合模型")
    else:
        raise ValueError(f"未知的模型類型: {args.model_type}")
    
    # 載入預訓練權重（如果存在）
    if args.model_path and os.path.exists(args.model_path):
        try:
            model.load_weights(args.model_path)
            logger.info(f"從 {args.model_path} 載入權重")
        except Exception as e:
            logger.error(f"載入權重失敗: {str(e)}")
    
    return model

def create_training_config(args):
    """創建訓練配置。"""
    config = {
        'batch_size': args.batch_size,
        'epochs_per_iteration': args.epochs,
        'num_self_play_games': args.num_games,
        'num_mcts_simulations': args.mcts_sims,
        'output_dir': args.output_dir,
        'log_dir': os.path.join(args.output_dir, 'logs'),
        'use_mixed_precision': args.use_mixed_precision,
        'lr_schedule': [(0, args.lr), (10000, args.lr/10), (50000, args.lr/100)]
    }
    
    logger.info(f"訓練配置: {config}")
    return config

def process_data(args):
    """處理SGF數據。"""
    logger.info(f"開始處理SGF數據，來源: {args.sgf_dir}")
    
    # 獲取所有SGF文件
    sgf_pattern = os.path.join(args.sgf_dir, '**/*.sgf')
    
    # 處理SGF文件
    train_data, val_data = process_sgf_files(
        sgf_pattern,
        output_file=args.data_file,
        min_rank=args.min_rank,
        verbose=args.verbose
    )
    
    logger.info(f"數據處理完成，保存到: {args.data_file}")
    return train_data, val_data

def selfplay(model, args):
    """執行自我對弈。"""
    logger.info(f"開始自我對弈，遊戲數: {args.num_games}")
    
    # 創建訓練配置
    config = create_training_config(args)
    
    # 創建訓練管線
    pipeline = TrainingPipeline(model, config=config)
    
    # 執行自我對弈
    states, policies, values = pipeline.self_play(num_games=args.num_games)
    
    # 保存自我對弈數據
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_file = os.path.join(args.output_dir, f'selfplay_{timestamp}.npz')
    np.savez_compressed(
        output_file,
        states=states,
        policies=policies,
        values=values
    )
    
    logger.info(f"自我對弈完成，數據已保存到: {output_file}")
    return states, policies, values

def train(model, args, initial_data=None):
    """訓練模型。"""
    logger.info(f"開始訓練，迭代次數: {args.iterations}")
    
    # 創建訓練配置
    config = create_training_config(args)
    
    # 創建訓練管線
    pipeline = TrainingPipeline(model, config=config)
    
    # 如果有初始數據，載入
    if initial_data is None and args.data_file and os.path.exists(args.data_file):
        train_data, val_data = load_data(args.data_file, verbose=args.verbose)
        initial_data = train_data
    
    # 執行訓練迭代
    pipeline.run_training(args.iterations, initial_data=initial_data)
    
    logger.info(f"訓練完成，模型已保存到: {args.output_dir}")
    return pipeline

def evaluate(model, args):
    """評估模型。"""
    logger.info("開始評估模型")
    
    # 載入對手模型（如果存在）
    opponent_model = None
    opponent_path = args.model_path + "_opponent" if args.model_path else None
    if opponent_path and os.path.exists(opponent_path):
        try:
            opponent_model = create_model(args)
            opponent_model.load_weights(opponent_path)
            logger.info(f"載入對手模型: {opponent_path}")
        except Exception as e:
            logger.error(f"載入對手模型失敗: {str(e)}")
    
    # 創建訓練配置
    config = create_training_config(args)
    
    # 創建訓練管線
    pipeline = TrainingPipeline(model, config=config)
    
    # 評估模型
    results = pipeline.evaluate_model(opponent_model=opponent_model, num_games=args.num_games)
    
    # 保存評估結果
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_file = os.path.join(args.output_dir, f'evaluation_{timestamp}.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    logger.info(f"評估完成，結果已保存到: {output_file}")
    return results

def main():
    """主函數。"""
    # 解析命令行參數
    args = parse_args()
    
    # 設置GPU選項
    set_gpu_options(args.gpu)
    
    # 確保輸出目錄存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 根據模式執行相應操作
    if args.mode == 'process_data':
        # 處理SGF數據
        train_data, val_data = process_data(args)
    else:
        # 創建模型
        model = create_model(args)
        
        if args.mode == 'selfplay':
            # 僅執行自我對弈
            states, policies, values = selfplay(model, args)
        elif args.mode == 'eval':
            # 評估模型
            results = evaluate(model, args)
        else:  # 'train'
            # 完整訓練
            pipeline = train(model, args)

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.exception(f"程序執行過程中發生錯誤: {str(e)}")
        sys.exit(1)