#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
圍棋AI訓練腳本 - 使用MCTS和CNN-RNN混合模型 (帶數據抽樣功能)
"""

import os
import sys
import tensorflow as tf
import numpy as np
import logging
from datetime import datetime
import argparse
from tqdm import tqdm

# 添加正確的模塊路徑
current_dir = os.path.dirname(os.path.abspath(__file__))
# 如果當前目錄不是項目根目錄，則添加項目根目錄
if os.path.basename(current_dir) != "SilvAGo":
    parent_dir = os.path.dirname(current_dir)
    sys.path.append(parent_dir)
else:
    sys.path.append(current_dir)

# 直接引入模塊，不使用相對導入
try:
    from CNN_RNN_A import create_cnn_rnn_model, GoGame
except ImportError:
    # 如果無法直接導入，嘗試讀取CNN_RNN_A.py文件的內容並執行
    print("嘗試直接載入CNN_RNN_A.py文件...")
    with open(os.path.join(current_dir, "CNN_RNN_A.py"), "r", encoding="utf-8") as f:
        code = f.read()
    exec(code)
    print("CNN_RNN_A.py已載入")

# 設置日誌記錄
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('train_cnn_mcts.log')
    ]
)
logger = logging.getLogger(__name__)

def set_cpu_only():
    """強制使用CPU進行計算。"""
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    logger.info("已強制使用CPU進行計算")

def load_and_preprocess_data(data_file, sequence_length=8, verbose=True, sample_rate=None, max_samples=None):
    """載入並預處理訓練數據，支持數據抽樣。
    
    Args:
        data_file: 數據文件路徑
        sequence_length: 序列長度
        verbose: 是否顯示詳細信息
        sample_rate: 抽樣比例 (0到1之間)
        max_samples: 最大樣本數量
        
    Returns:
        train_data, val_data: 訓練集和驗證集
    """
    if verbose:
        logger.info(f"從 {data_file} 載入數據...")
    
    try:
        data = np.load(data_file)
        
        # 讀取棋盤狀態和下一步移動
        X = data['X']
        y = data['y']
        
        # 檢查是否有勝負信息
        has_winners = 'winners' in data
        if has_winners and verbose:
            logger.info("找到勝負信息！使用實際比賽結果作為價值標籤")
            winners = data['winners']
        else:
            if verbose:
                logger.info("未找到勝負信息，將使用啟發式方法估計價值")
        
        total_samples = len(X)
        if verbose:
            logger.info(f"原始數據形狀：X: {X.shape}, y: {y.shape}")
            if has_winners:
                logger.info(f"winners: {winners.shape}")
        
        # 應用數據抽樣
        if sample_rate is not None or max_samples is not None:
            # 確定要保留的樣本數量
            if max_samples is not None:
                num_samples = min(total_samples, max_samples)
            elif sample_rate is not None:
                num_samples = int(total_samples * sample_rate)
            else:
                num_samples = total_samples
            
            # 隨機選擇樣本索引
            indices = np.random.choice(total_samples, num_samples, replace=False)
            indices.sort()  # 排序以保持序列的連續性
            
            X = X[indices]
            y = y[indices]
            if has_winners:
                winners = winners[indices]
            
            if verbose:
                logger.info(f"應用抽樣後的數據形狀：X: {X.shape}, y: {y.shape}")
                logger.info(f"抽樣比例: {len(X)/total_samples:.2%}")
        
        # 預分配數組
        X_board = []
        X_sequence = []
        y_policy = []
        y_value = []
        
        # 使用tqdm添加進度條
        if verbose:
            logger.info("開始處理數據...")
            iterator = tqdm(range(len(X)), desc="處理樣本")
        else:
            iterator = range(len(X))
        
        # 處理棋盤特徵
        for i in iterator:
            # 創建棋盤特徵平面 (19x19x3)
            features = np.zeros((19, 19, 3), dtype=np.float32)
            
            # 當前棋盤狀態 (黑子和白子)
            features[:, :, 0] = (X[i] == 1)  # 黑子
            features[:, :, 1] = (X[i] == -1)  # 白子
            
            # 當前玩家 (從棋子數量推斷)
            black_count = np.sum(X[i] == 1)
            white_count = np.sum(X[i] == -1)
            current_player = 1 if black_count <= white_count else -1
            features[:, :, 2] = current_player  # 當前玩家
            
            X_board.append(features)
            
            # 處理序列特徵
            # 為每個位置創建序列數據（之前的棋盤狀態）
            start_idx = max(0, i - sequence_length + 1)
            sequence = []
            for j in range(start_idx, i + 1):
                # 查找原始索引
                orig_idx = indices[j] if (sample_rate is not None or max_samples is not None) else j
                
                # 為每個歷史棋盤創建特徵
                flat_features = np.zeros(19*19*3, dtype=np.float32)
                flat_features[:19*19] = (X[j] == 1).flatten()  # 黑子
                flat_features[19*19:2*19*19] = (X[j] == -1).flatten()  # 白子
                
                # 當前玩家
                b_count = np.sum(X[j] == 1)
                w_count = np.sum(X[j] == -1)
                curr_player = 1 if b_count <= w_count else -1
                flat_features[2*19*19:] = curr_player
                
                sequence.append(flat_features)
            
            # 填充序列至指定長度
            while len(sequence) < sequence_length:
                sequence.insert(0, np.zeros(19*19*3, dtype=np.float32))
            
            X_sequence.append(sequence)
            
            # 處理策略標籤 (下一步移動)
            policy = np.zeros(19*19, dtype=np.float32)
            move_positions = np.where(y[i] == 1)
            if len(move_positions[0]) > 0:
                move_idx = move_positions[0][0] * 19 + move_positions[1][0]
                policy[move_idx] = 1.0
            y_policy.append(policy)
            
            # 處理價值標籤
            if has_winners:
                # 使用實際勝負信息
                y_value.append([winners[i]])
            else:
                # 使用啟發式方法估計價值
                black_stones = np.sum(X[i] == 1)
                white_stones = np.sum(X[i] == -1)
                if black_stones + white_stones > 0:
                    # 黑白棋子數量差異，歸一化到 [-0.5, 0.5] 範圍
                    board_control = 0.5 * (black_stones - white_stones) / (black_stones + white_stones)
                    # 從當前玩家的角度來看價值
                    value = board_control * current_player
                else:
                    value = 0.0
                    
                y_value.append([value])
        
        # 轉換為NumPy數組
        X_board = np.array(X_board, dtype=np.float32)
        X_sequence = np.array(X_sequence, dtype=np.float32)
        y_policy = np.array(y_policy, dtype=np.float32)
        y_value = np.array(y_value, dtype=np.float32)
        
        # 分割訓練集和驗證集 (9:1)
        split_idx = int(0.9 * len(X_board))
        
        train_data = {
            'X_board': X_board[:split_idx],
            'X_sequence': X_sequence[:split_idx],
            'y_policy': y_policy[:split_idx],
            'y_value': y_value[:split_idx]
        }
        
        val_data = {
            'X_board': X_board[split_idx:],
            'X_sequence': X_sequence[split_idx:],
            'y_policy': y_policy[split_idx:],
            'y_value': y_value[split_idx:]
        }
        
        if verbose:
            logger.info(f"處理完成。訓練集大小: {len(train_data['X_board'])}, 驗證集大小: {len(val_data['X_board'])}")
            logger.info(f"價值標籤範圍: min={np.min(y_value):.4f}, max={np.max(y_value):.4f}, mean={np.mean(y_value):.4f}")
        
        return train_data, val_data
    
    except Exception as e:
        logger.error(f"載入數據時出錯: {str(e)}")
        raise

def train_cnn_rnn_model(data_file, output_dir, epochs=30, batch_size=32, sample_rate=None, max_samples=None):
    """訓練CNN-RNN模型"""
    # 建立輸出目錄
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'logs'), exist_ok=True)
    
    # 載入並預處理數據
    train_data, val_data = load_and_preprocess_data(
        data_file, 
        verbose=True, 
        sample_rate=sample_rate, 
        max_samples=max_samples
    )
    
    # 創建CNN-RNN模型
    logger.info("創建CNN-RNN混合模型...")
    model = create_cnn_rnn_model(input_shape=(19, 19, 3), sequence_length=8)
    
    # 重新編譯模型，調整損失權重
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss={'policy': 'categorical_crossentropy', 'value': 'mse'},
        loss_weights={'policy': 1.0, 'value': 1.0},  # 可以根據需要調整權重
        metrics={'policy': 'accuracy', 'value': 'mae'}
    )
    
    model.summary()  # 打印模型結構
    
    # 設置回調函數
    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir=os.path.join(output_dir, 'logs')),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(output_dir, 'checkpoint_{epoch:02d}.keras'),
            save_best_only=True,
            monitor='val_loss'
        ),
        tf.keras.callbacks.EarlyStopping(
            patience=5,
            restore_best_weights=True,
            monitor='val_loss'
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            factor=0.5,
            patience=3,
            monitor='val_loss'
        )
    ]
    
    # 訓練模型
    logger.info("開始訓練模型...")
    history = model.fit(
        [train_data['X_board'], train_data['X_sequence']],
        {'policy': train_data['y_policy'], 'value': train_data['y_value']},
        validation_data=(
            [val_data['X_board'], val_data['X_sequence']],
            {'policy': val_data['y_policy'], 'value': val_data['y_value']}
        ),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks
    )
    
    # 保存最終模型
    model_path = os.path.join(output_dir, 'final_model.keras')
    model.save(model_path)
    logger.info(f"訓練完成，模型已保存到 {model_path}")
    
    # 繪製訓練歷史
    plot_training_history(history, output_dir)
    
    return model, history

def plot_training_history(history, output_dir):
    """繪製訓練歷史曲線"""
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(15, 5))
        
        # 損失曲線
        plt.subplot(1, 3, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # 策略準確率
        plt.subplot(1, 3, 2)
        plt.plot(history.history['policy_accuracy'], label='Policy Accuracy')
        plt.plot(history.history['val_policy_accuracy'], label='Val Policy Accuracy')
        plt.title('Policy Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # 價值MAE
        plt.subplot(1, 3, 3)
        plt.plot(history.history['value_mae'], label='Value MAE')
        plt.plot(history.history['val_value_mean_absolute_error'], label='Val Value MAE')
        plt.title('Value Mean Absolute Error')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'training_history.png'))
        plt.close()
        logger.info(f"訓練歷史圖表已保存到 {os.path.join(output_dir, 'training_history.png')}")
    except Exception as e:
        logger.error(f"繪製訓練歷史時出錯: {str(e)}")

def parse_args():
    """解析命令行參數"""
    parser = argparse.ArgumentParser(description='圍棋AI訓練腳本')
    
    # 數據相關
    parser.add_argument('--data_file', type=str, default=r"C:\Users\User\Desktop\SilvAGo\process_kgs_dataset_10.npz",
                       help='數據文件路徑')
    parser.add_argument('--sample_rate', type=float, default=0.4,
                       help='數據抽樣比例 (0到1之間)')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='最大樣本數量')
    
    # 模型相關
    parser.add_argument('--output_dir', type=str, default='models/cnn_rnn_mcts',
                       help='模型輸出目錄')
    
    # 訓練參數
    parser.add_argument('--epochs', type=int, default=30,
                       help='訓練輪數')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批次大小')
    
    # CPU/GPU選擇
    parser.add_argument('--use_gpu', action='store_true',
                       help='是否使用GPU (默認使用CPU)')
    
    return parser.parse_args()

def main():
    # 解析命令行參數
    args = parse_args()
    
    # 設置CPU/GPU
    if not args.use_gpu:
        set_cpu_only()
    
    # 訓練模型
    model, history = train_cnn_rnn_model(
        data_file=args.data_file,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        sample_rate=args.sample_rate,
        max_samples=args.max_samples
    )
    
    logger.info("訓練過程已完成")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception(f"程序執行過程中發生錯誤: {str(e)}")
        sys.exit(1)