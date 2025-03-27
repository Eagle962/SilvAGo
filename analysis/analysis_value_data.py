import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import random

# 設置日誌
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def diagnose_value_prediction(data_file, output_dir='value_diagnosis', sample_rate=0.05):
    """診斷價值預測問題並嘗試單獨訓練價值網絡
    
    Args:
        data_file: 數據文件路徑
        output_dir: 輸出目錄
        sample_rate: 數據抽樣比例 (0.05 表示使用5%的數據)
    """
    os.makedirs(output_dir, exist_ok=True)
    
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
    
    # 數據抽樣
    total_samples = len(X)
    sample_size = int(total_samples * sample_rate)
    logger.info(f"原始數據大小: {total_samples}，抽樣比例: {sample_rate}，抽樣後大小: {sample_size}")
    
    # 隨機選擇索引
    indices = random.sample(range(total_samples), sample_size)
    indices.sort()  # 排序以保持連續性
    
    X = X[indices]
    y = y[indices]
    if has_winners:
        winners = winners[indices]
    
    logger.info(f"抽樣後的數據形狀: X: {X.shape}, y: {y.shape}")
    
    # 處理數據 (簡化版)
    X_board = []
    y_value = []
    
    for i in tqdm(range(len(X)), desc="處理樣本"):
        # 簡化特徵 - 只使用棋盤狀態，不使用序列
        features = np.zeros((19, 19, 3), dtype=np.float32)
        features[:, :, 0] = (X[i] == 1)  # 黑子
        features[:, :, 1] = (X[i] == -1)  # 白子
        
        # 當前玩家
        black_count = np.sum(X[i] == 1)
        white_count = np.sum(X[i] == -1)
        current_player = 1 if black_count <= white_count else -1
        features[:, :, 2] = current_player
        
        X_board.append(features)
        
        # 處理價值標籤
        if has_winners:
            y_value.append([winners[i] * current_player])
        else:
            # 簡單估計
            black_stones = np.sum(X[i] == 1)
            white_stones = np.sum(X[i] == -1)
            if black_stones + white_stones > 0:
                board_control = 0.5 * (black_stones - white_stones) / (black_stones + white_stones)
                value = board_control * current_player
            else:
                value = 0.0
            y_value.append([value])
    
    # 轉換為NumPy數組
    X_board = np.array(X_board, dtype=np.float32)
    y_value = np.array(y_value, dtype=np.float32)
    
    # 分析價值標籤
    logger.info(f"價值標籤統計: min={np.min(y_value):.4f}, max={np.max(y_value):.4f}, mean={np.mean(y_value):.4f}, std={np.std(y_value):.4f}")
    
    # 繪製價值分佈
    plt.figure(figsize=(10, 6))
    plt.hist(y_value, bins=50)
    plt.title('價值標籤分佈')
    plt.xlabel('價值')
    plt.ylabel('頻率')
    plt.savefig(os.path.join(output_dir, 'value_distribution.png'))
    plt.close()
    
    # 分割訓練集和驗證集
    split_idx = int(0.9 * len(X_board))
    
    X_train, X_val = X_board[:split_idx], X_board[split_idx:]
    y_train, y_val = y_value[:split_idx], y_value[split_idx:]
    
    # 驗證集分析
    logger.info(f"驗證集價值標籤統計: min={np.min(y_val):.4f}, max={np.max(y_val):.4f}, mean={np.mean(y_val):.4f}, std={np.std(y_val):.4f}")
    
    # 繪製驗證集價值分佈
    plt.figure(figsize=(10, 6))
    plt.hist(y_val, bins=50)
    plt.title('驗證集價值標籤分佈')
    plt.xlabel('價值')
    plt.ylabel('頻率')
    plt.savefig(os.path.join(output_dir, 'val_value_distribution.png'))
    plt.close()
    
    # 創建純價值網絡
    def create_value_only_model():
        inputs = tf.keras.layers.Input(shape=(19, 19, 3))
        
        # 使用更簡單的卷積網絡
        x = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
        
        x = tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        outputs = tf.keras.layers.Dense(1, activation='tanh')(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
            loss='mse',
            metrics=['mae', tf.keras.metrics.RootMeanSquaredError()]
        )
        return model
    
    # 創建和訓練純價值網絡
    logger.info("創建並訓練純價值網絡...")
    value_model = create_value_only_model()
    value_model.summary()
    
    # 設置回調
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(output_dir, 'best_value_model.keras'),
            save_best_only=True
        ),
        # 使用高初始學習率並逐漸降低
        tf.keras.callbacks.LearningRateScheduler(
            lambda epoch, lr: 0.01 * (0.8 ** epoch)
        )
    ]
    
    # 縮短訓練時間
    epochs = 20  # 減少訓練輪數
    
    # 訓練模型
    history = value_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=64,
        callbacks=callbacks,
        verbose=1
    )
    
    # 繪製訓練歷史
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Value Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Value Model MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'value_model_history.png'))
    plt.close()
    
    # 評估模型
    logger.info("評估價值模型...")
    results = value_model.evaluate(X_val, y_val, verbose=1)
    for name, value in zip(value_model.metrics_names, results):
        logger.info(f"{name}: {value:.4f}")
    
    # 預測並分析
    y_pred = value_model.predict(X_val)
    
    # 繪製預測與真實值的散點圖
    plt.figure(figsize=(10, 10))
    plt.scatter(y_val, y_pred, alpha=0.5)
    plt.plot([-1, 1], [-1, 1], 'r--')
    plt.xlabel('真實價值')
    plt.ylabel('預測價值')
    plt.title('價值預測散點圖')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'value_predictions.png'))
    plt.close()
    
    logger.info("診斷完成，結果已保存到 " + output_dir)
    return value_model, history

# 運行診斷
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        data_file = sys.argv[1]
    else:
        data_file = "process_kgs_dataset_50000.npz"  
    
    # 使用5%的數據進行測試
    diagnose_value_prediction(data_file, sample_rate=0.05)