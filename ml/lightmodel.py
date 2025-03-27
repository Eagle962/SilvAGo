import tensorflow as tf
from tensorflow.keras import layers, models

def create_lightweight_cnn_rnn_model(input_shape=(19, 19, 3), sequence_length=8):
    """創建一個輕量級的CNN-RNN混合模型，適合較小的數據集。
    
    Args:
        input_shape: 棋盤輸入的形狀，默認為(19, 19, 3)
        sequence_length: 序列長度，默認為8
        
    Returns:
        編譯好的Keras模型
    """
    # 棋盤輸入
    board_input = layers.Input(shape=input_shape, name='input_layer')
    
    # 初始卷積層 (減少濾波器數量)
    x = layers.Conv2D(32, (3, 3), padding='same')(board_input)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    # 只使用2個殘差塊而不是原來的5個
    def residual_block(x, filters):
        shortcut = x
        x = layers.Conv2D(filters, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv2D(filters, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Add()([shortcut, x])
        x = layers.ReLU()(x)
        return x
    
    # 添加較少的殘差塊
    for _ in range(2):  # 原來是5個
        x = residual_block(x, 32)  # 原來是64
    
    # 特徵提取
    cnn_features = layers.Conv2D(64, (3, 3), padding='same')(x)  # 原來是128
    cnn_features = layers.BatchNormalization()(cnn_features)
    cnn_features = layers.ReLU()(cnn_features)
    
    # 序列輸入 (減少維度)
    sequence_input = layers.Input(shape=(sequence_length, 19*19*3), name='input_layer_1')
    
    # 使用單層LSTM而不是雙層
    rnn = layers.LSTM(128)(sequence_input)  # 原來是兩層LSTM，第一層256單元
    
    # 合併特徵
    cnn_flat = layers.Flatten()(cnn_features)
    combined = layers.Concatenate()([cnn_flat, rnn])
    
    # 策略頭 (減少神經元)
    policy = layers.Dense(256, activation='relu')(combined)  # 原來是512
    policy = layers.Dropout(0.3)(policy)
    policy_output = layers.Dense(361, activation='softmax', name='policy')(policy)
    
    # 價值頭 (減少神經元)
    value = layers.Dense(128, activation='relu')(combined)  # 原來是256
    value = layers.Dropout(0.3)(value)
    value_output = layers.Dense(1, activation='tanh', name='value')(value)
    
    # 創建模型
    model = models.Model(
        inputs=[board_input, sequence_input],
        outputs=[policy_output, value_output]
    )
    
    # 編譯模型
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss={
            'policy': 'categorical_crossentropy',
            'value': 'mse'
        },
        metrics={
            'policy': 'accuracy',
            'value': 'mae'
        }
    )
    
    return model

def train_lightweight_model(data_file, output_dir, epochs=30, batch_size=64, sample_rate=0.1, max_samples=None):
    """訓練輕量級CNN-RNN圍棋模型
    
    Args:
        data_file: 數據文件路徑
        output_dir: 輸出目錄
        epochs: 訓練輪數
        batch_size: 批次大小
        sample_rate: 數據抽樣比例
        max_samples: 最大樣本數量
    
    Returns:
        model: 訓練好的模型
        history: 訓練歷史
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    import logging
    
    # 設置日誌
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
    )
    logger = logging.getLogger(__name__)
    
    # 確保輸出目錄存在
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'logs'), exist_ok=True)
    
    # 載入數據
    logger.info(f"從 {data_file} 載入數據...")
    data = np.load(data_file)
    X = data['X']
    y = data['y']
    
    # 檢查是否有勝負信息
    has_winners = 'winners' in data
    if has_winners:
        logger.info("找到勝負信息！使用實際比賽結果作為價值標籤")
        winners = data['winners']
    else:
        logger.info("未找到勝負信息，將使用啟發式方法估計價值")
    
    total_samples = len(X)
    logger.info(f"原始數據形狀：X: {X.shape}, y: {y.shape}")
    
    # 應用數據抽樣
    if sample_rate is not None or max_samples is not None:
        if max_samples is not None:
            num_samples = min(total_samples, max_samples)
        elif sample_rate is not None:
            num_samples = int(total_samples * sample_rate)
        
        # 隨機選擇樣本
        indices = np.random.choice(total_samples, num_samples, replace=False)
        indices.sort()  # 排序以保持序列的連續性
        
        X = X[indices]
        y = y[indices]
        if has_winners:
            winners = winners[indices]
        
        logger.info(f"應用抽樣後的數據形狀：X: {X.shape}, y: {y.shape}")
        logger.info(f"抽樣比例: {len(X)/total_samples:.2%}")
    
    # 預處理數據
    logger.info("開始處理數據...")
    X_board = []
    X_sequence = []
    y_policy = []
    y_value = []
    
    sequence_length = 8  # 定義序列長度
    
    # 使用tqdm添加進度條
    for i in tqdm(range(len(X)), desc="處理樣本"):
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
            y_value.append([winners[i] * current_player])
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
    
    # 顯示價值標籤統計
    logger.info(f"價值標籤範圍: min={np.min(y_value):.4f}, max={np.max(y_value):.4f}, mean={np.mean(y_value):.4f}")
    
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
    
    logger.info(f"處理完成。訓練集大小: {len(train_data['X_board'])}, 驗證集大小: {len(val_data['X_board'])}")
    
    # 創建輕量級模型
    logger.info("創建輕量級CNN-RNN混合模型...")
    model = create_lightweight_cnn_rnn_model(input_shape=(19, 19, 3), sequence_length=sequence_length)
    model.summary()
    
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
    model_path = os.path.join(output_dir, 'final_lightweight_model.keras')
    model.save(model_path)
    logger.info(f"訓練完成，模型已保存到 {model_path}")
    
    # 繪製訓練歷史
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
    plt.plot(history.history['val_value_mae'], label='Val Value MAE')
    plt.title('Value Mean Absolute Error')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'))
    plt.close()
    logger.info(f"訓練歷史圖表已保存到 {os.path.join(output_dir, 'training_history.png')}")
    
    return model, history

if __name__ == "__main__":
    import argparse
    
    # 解析命令行參數
    parser = argparse.ArgumentParser(description='訓練輕量級圍棋AI模型')
    
    # 數據相關
    parser.add_argument('--data_file', type=str, required=True,
                       help='數據文件路徑')
    parser.add_argument('--output_dir', type=str, default='models/lightweight_go',
                       help='模型輸出目錄')
    parser.add_argument('--sample_rate', type=float, default=0.1,
                       help='數據抽樣比例 (0到1之間)')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='最大樣本數量')
    
    # 訓練參數
    parser.add_argument('--epochs', type=int, default=30,
                       help='訓練輪數')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='批次大小')
    
    # 解析參數
    args = parser.parse_args()
    
    # 訓練模型
    train_lightweight_model(
        data_file=args.data_file,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        sample_rate=args.sample_rate,
        max_samples=args.max_samples
    )