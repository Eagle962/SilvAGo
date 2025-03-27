import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

def create_enhanced_cnn_rnn_model(input_shape=(19, 19, 3), sequence_length=8):
    """創建一個增強版的CNN-RNN混合模型，專注於提升價值預測能力。
    
    Args:
        input_shape: 棋盤輸入的形狀，默認為(19, 19, 3)
        sequence_length: 序列長度，默認為8
        
    Returns:
        編譯好的Keras模型
    """
    # 棋盤輸入
    board_input = layers.Input(shape=input_shape, name='input_layer')
    
    # 初始卷積層 (增加濾波器數量以提高特徵表示能力)
    x = layers.Conv2D(48, (3, 3), padding='same', 
                     kernel_regularizer=regularizers.l2(1e-4))(board_input)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    # 創建改進的殘差塊（添加正則化）
    def residual_block(x, filters):
        shortcut = x
        x = layers.Conv2D(filters, (3, 3), padding='same',
                         kernel_regularizer=regularizers.l2(1e-4))(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv2D(filters, (3, 3), padding='same',
                         kernel_regularizer=regularizers.l2(1e-4))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Add()([shortcut, x])
        x = layers.ReLU()(x)
        return x
    
    # 添加適量的殘差塊
    for _ in range(3):  # 增加到3個殘差塊
        x = residual_block(x, 48)  # 增加到48個濾波器
    
    # 特徵提取
    cnn_features = layers.Conv2D(96, (3, 3), padding='same',
                                kernel_regularizer=regularizers.l2(1e-4))(x)  # 增加到96
    cnn_features = layers.BatchNormalization()(cnn_features)
    cnn_features = layers.ReLU()(cnn_features)
    
    # 序列輸入
    sequence_input = layers.Input(shape=(sequence_length, 19*19*3), name='input_layer_1')
    
    # 改進的LSTM處理，使用雙向LSTM提高時序特徵的捕捉能力
    rnn = layers.Bidirectional(layers.LSTM(128, return_sequences=True, 
                                          recurrent_regularizer=regularizers.l2(1e-4)))(sequence_input)
    rnn = layers.Bidirectional(layers.LSTM(128, 
                                          recurrent_regularizer=regularizers.l2(1e-4)))(rnn)
    
    # 合併特徵
    cnn_flat = layers.Flatten()(cnn_features)
    combined = layers.Concatenate()([cnn_flat, rnn])
    shared = layers.Dense(512, activation='relu', 
                         kernel_regularizer=regularizers.l2(1e-4))(combined)
    shared = layers.Dropout(0.4)(shared)  # 增加dropout率
    
    # 策略頭
    policy = layers.Dense(256, activation='relu', 
                         kernel_regularizer=regularizers.l2(1e-4))(shared)
    policy = layers.Dropout(0.3)(policy)
    policy_output = layers.Dense(361, activation='softmax', name='policy',
                                kernel_regularizer=regularizers.l2(1e-4))(policy)
    
    # 增強的價值頭 - 使用更深的網絡結構
    value = layers.Dense(256, activation='relu', 
                        kernel_regularizer=regularizers.l2(1e-5))(shared)
    value = layers.Dropout(0.4)(value)
    value = layers.Dense(128, activation='relu', 
                        kernel_regularizer=regularizers.l2(1e-5))(value)
    value = layers.Dropout(0.4)(value)
    value = layers.Dense(64, activation='relu', 
                        kernel_regularizer=regularizers.l2(1e-5))(value)
    value = layers.Dropout(0.3)(value)
    value_output = layers.Dense(1, activation='tanh', name='value',
                               kernel_regularizer=regularizers.l2(1e-5))(value)
    
    # 創建模型
    model = models.Model(
        inputs=[board_input, sequence_input],
        outputs=[policy_output, value_output]
    )
    
    # 編譯模型 - 使用更高的學習率和值損失權重
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),  # 更高的初始學習率
        loss={
            'policy': 'categorical_crossentropy',
            'value': 'mse'
        },
        loss_weights={
            'policy': 1.0,
            'value': 3.0  # 進一步增加價值損失的權重
        },
        metrics={
            'policy': 'accuracy',
            'value': ['mae', tf.keras.metrics.RootMeanSquaredError()]  # 添加RMSE指標
        }
    )
    
    return model

# 創建策略頭和價值頭使用不同優化器的模型
def create_dual_optimizer_model(board_input, sequence_input, policy_output, value_output):
    """創建使用雙優化器的模型 - 為策略頭和價值頭使用不同的學習率"""
    from tensorflow.keras import backend as K
    
    # 創建基本模型
    model = models.Model(
        inputs=[board_input, sequence_input],
        outputs=[policy_output, value_output]
    )
    
    # 為兩個頭部創建不同的優化器
    policy_optimizer = tf.keras.optimizers.Adam(learning_rate=0.005)
    value_optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)  # 價值頭使用更高的學習率
    
    # 獲取模型所有層
    policy_layers = [layer for layer in model.layers 
                      if ('policy' in layer.name or 
                          isinstance(layer, layers.Dense) and 'shared' not in layer.name)]
    
    value_layers = [layer for layer in model.layers 
                     if ('value' in layer.name or 
                         isinstance(layer, layers.Dense) and 'shared' not in layer.name)]
    
    # 為不同的層使用不同的優化器
    policy_trainable_weights = []
    value_trainable_weights = []
    
    for layer in policy_layers:
        policy_trainable_weights.extend(layer.trainable_weights)
    
    for layer in value_layers:
        value_trainable_weights.extend(layer.trainable_weights)
    
    # 自定義訓練步驟
    @tf.function
    def train_step(data):
        inputs, targets = data
        
        with tf.GradientTape() as policy_tape, tf.GradientTape() as value_tape:
            # 前向傳播
            outputs = model(inputs, training=True)
            
            # 計算損失
            policy_loss = tf.keras.losses.categorical_crossentropy(targets['policy'], outputs[0])
            value_loss = tf.keras.losses.mean_squared_error(targets['value'], outputs[1])
            
            # 應用權重
            policy_loss = 1.0 * tf.reduce_mean(policy_loss)
            value_loss = 3.0 * tf.reduce_mean(value_loss)
            
            # 添加正則化損失
            policy_loss += sum(model.losses)
        
        # 計算梯度
        policy_gradients = policy_tape.gradient(policy_loss, policy_trainable_weights)
        value_gradients = value_tape.gradient(value_loss, value_trainable_weights)
        
        # 應用梯度
        policy_optimizer.apply_gradients(zip(policy_gradients, policy_trainable_weights))
        value_optimizer.apply_gradients(zip(value_gradients, value_trainable_weights))
        
        # 更新度量
        model.compiled_metrics.update_state(targets, outputs)
        
        # 返回度量
        results = {m.name: m.result() for m in model.metrics}
        results['policy_loss'] = policy_loss
        results['value_loss'] = value_loss
        return results
    
    # 設置自定義訓練步驟
    model.train_step = train_step
    
    # 編譯模型
    model.compile(
        optimizer=policy_optimizer,  # 這個會被自定義訓練步驟覆蓋
        loss={
            'policy': 'categorical_crossentropy',
            'value': 'mse'
        },
        metrics={
            'policy': 'accuracy',
            'value': ['mae', tf.keras.metrics.RootMeanSquaredError()]
        }
    )
    
    return model

def train_enhanced_model(data_file, output_dir, epochs=30, batch_size=64, sample_rate=0.1, max_samples=None):
    """訓練增強型CNN-RNN圍棋模型，專注於價值預測能力
    
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
            # 使用更複雜的啟發式方法估計價值
            # 考慮棋盤控制率和棋子連通性
            black_stones = np.sum(X[i] == 1)
            white_stones = np.sum(X[i] == -1)
            
            if black_stones + white_stones > 0:
                # 基本控制率
                board_control = 0.6 * (black_stones - white_stones) / (black_stones + white_stones)
                
                # 計算簡單的棋子連通性（連續的相同顏色棋子）
                connectivity_score = 0
                for r in range(X[i].shape[0]):
                    for c in range(X[i].shape[1]):
                        if X[i][r, c] == current_player:
                            # 檢查四個方向
                            neighbors = 0
                            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                                nr, nc = r + dr, c + dc
                                if (0 <= nr < X[i].shape[0] and 
                                    0 <= nc < X[i].shape[1] and 
                                    X[i][nr, nc] == current_player):
                                    neighbors += 1
                            connectivity_score += neighbors / 4.0  # 歸一化到0-1
                
                if black_stones + white_stones > 5:  # 只有在棋盤上有足夠的棋子時才考慮連通性
                    stones_of_current_player = black_stones if current_player == 1 else white_stones
                    if stones_of_current_player > 0:
                        connectivity_score = 0.4 * (connectivity_score / stones_of_current_player)
                        value = board_control + connectivity_score
                    else:
                        value = board_control
                else:
                    value = board_control
                
                # 限制在[-1, 1]範圍內
                value = max(-1.0, min(1.0, value)) * current_player
            else:
                value = 0.0
                
            y_value.append([value])
    
    # 轉換為NumPy數組
    X_board = np.array(X_board, dtype=np.float32)
    X_sequence = np.array(X_sequence, dtype=np.float32)
    y_policy = np.array(y_policy, dtype=np.float32)
    y_value = np.array(y_value, dtype=np.float32)
    
    # 檢查價值標籤分佈並顯示
    logger.info(f"價值標籤統計: min={np.min(y_value):.4f}, max={np.max(y_value):.4f}, mean={np.mean(y_value):.4f}, std={np.std(y_value):.4f}")
    
    # 顯示價值標籤直方圖
    plt.figure(figsize=(10, 6))
    plt.hist(y_value, bins=50)
    plt.title('價值標籤分佈')
    plt.xlabel('價值')
    plt.ylabel('頻率')
    plt.savefig(os.path.join(output_dir, 'value_distribution.png'))
    plt.close()
    logger.info(f"價值標籤分佈圖已保存到 {os.path.join(output_dir, 'value_distribution.png')}")
    
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
    
    # 檢查驗證集價值標籤統計
    logger.info(f"驗證集價值標籤統計: min={np.min(val_data['y_value']):.4f}, " 
                f"max={np.max(val_data['y_value']):.4f}, "
                f"mean={np.mean(val_data['y_value']):.4f}, "
                f"std={np.std(val_data['y_value']):.4f}")
    
    logger.info(f"處理完成。訓練集大小: {len(train_data['X_board'])}, 驗證集大小: {len(val_data['X_board'])}")
    
    # 創建增強型模型
    logger.info("創建增強型CNN-RNN混合模型...")
    model = create_enhanced_cnn_rnn_model(input_shape=(19, 19, 3), sequence_length=sequence_length)
    model.summary()
    
    # 設置自定義回調函數來檢測價值頭的變化
    class ValueMonitorCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            # 計算驗證集上的價值頭性能
            val_values_pred = self.model.predict([val_data['X_board'], val_data['X_sequence']])[1]
            val_values_true = val_data['y_value']
            
            # 計算簡單統計
            mean_pred = np.mean(val_values_pred)
            std_pred = np.std(val_values_pred)
            min_pred = np.min(val_values_pred)
            max_pred = np.max(val_values_pred)
            
            # 記錄價值頭預測統計
            logs['val_value_mean'] = mean_pred
            logs['val_value_std'] = std_pred
            logs['val_value_min'] = min_pred
            logs['val_value_max'] = max_pred
            
            logger.info(f"Epoch {epoch+1} - 價值頭預測統計: "
                       f"mean={mean_pred:.4f}, std={std_pred:.4f}, "
                       f"min={min_pred:.4f}, max={max_pred:.4f}")
    
    # 設置回調函數
    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir=os.path.join(output_dir, 'logs'),
                                      histogram_freq=1),  # 啟用直方圖
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(output_dir, 'checkpoint_{epoch:02d}.keras'),
            save_best_only=True,
            monitor='val_loss'
        ),
        tf.keras.callbacks.EarlyStopping(
            patience=7,  # 增加耐心
            restore_best_weights=True,
            monitor='val_loss'
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            factor=0.5,
            patience=3,
            monitor='val_loss',
            min_lr=0.0001
        ),
        # 使用循環學習率
        tf.keras.callbacks.LearningRateScheduler(
            lambda epoch, lr: 0.01 * (0.5 ** (epoch // 5))  # 每5個epoch將學習率減半
        ),
        # 自定義價值監視器
        ValueMonitorCallback()
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
        callbacks=callbacks,
        shuffle=True  # 確保每個epoch都打亂數據
    )
    
    # 保存最終模型
    model_path = os.path.join(output_dir, 'final_enhanced_model.keras')
    model.save(model_path)
    logger.info(f"訓練完成，模型已保存到 {model_path}")
    
    # 繪製訓練歷史
    plt.figure(figsize=(15, 10))
    
    # 損失曲線
    plt.subplot(2, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # 策略準確率
    plt.subplot(2, 2, 2)
    plt.plot(history.history['policy_accuracy'], label='Policy Accuracy')
    plt.plot(history.history['val_policy_accuracy'], label='Val Policy Accuracy')
    plt.title('Policy Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # 價值MAE
    plt.subplot(2, 2, 3)
    plt.plot(history.history['value_mae'], label='Value MAE')
    plt.plot(history.history['val_value_mae'], label='Val Value MAE')
    plt.title('Value Mean Absolute Error')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    
    # 損失細分
    plt.subplot(2, 2, 4)
    plt.plot(history.history['policy_loss'], label='Policy Loss')
    plt.plot(history.history['value_loss'], label='Value Loss')
    plt.plot(history.history['val_policy_loss'], label='Val Policy Loss')
    plt.plot(history.history['val_value_loss'], label='Val Value Loss')
    plt.title('Loss Components')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'))
    plt.close()
    logger.info(f"訓練歷史圖表已保存到 {os.path.join(output_dir, 'training_history.png')}")
    
    # 評估模型
    logger.info("評估模型...")
    eval_results = model.evaluate(
        [val_data['X_board'], val_data['X_sequence']],
        {'policy': val_data['y_policy'], 'value': val_data['y_value']},
        verbose=1
    )
    
    # 獲取結果
    metrics = model.metrics_names
    for name, value in zip(metrics, eval_results):
        logger.info(f"{name}: {value:.4f}")
    
    # 預測並分析價值
    y_pred = model.predict([val_data['X_board'], val_data['X_sequence']])[1]
    
    # 繪製預測VS真實價值的散點圖
    plt.figure(figsize=(10, 10))
    plt.scatter(val_data['y_value'], y_pred, alpha=0.5)
    plt.plot([-1, 1], [-1, 1], 'r--')
    plt.xlabel('真實價值')
    plt.ylabel('預測價值')
    plt.title('價值預測散點圖')
    plt.xlim([-1.1, 1.1])
    plt.ylim([-1.1, 1.1])
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'value_prediction.png'))
    plt.close()
    logger.info(f"價值預測散點圖已保存到 {os.path.join(output_dir, 'value_prediction.png')}")
    
    return model, history

if __name__ == "__main__":
    import argparse
    
    # 解析命令行參數
    parser = argparse.ArgumentParser(description='訓練增強型圍棋AI模型')
    
    # 數據相關
    parser.add_argument('--data_file', type=str, required=True,
                       help='數據文件路徑')
    parser.add_argument('--output_dir', type=str, default='models/enhanced_go',
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
    train_enhanced_model(
        data_file=args.data_file,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        sample_rate=args.sample_rate,
        max_samples=args.max_samples
    )