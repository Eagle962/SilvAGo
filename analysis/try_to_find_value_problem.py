import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import random
import time

# 設置日誌
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PredictionCheckCallback(tf.keras.callbacks.Callback):
    """監控每個epoch的預測變化"""
    def __init__(self, validation_data):
        super().__init__()
        self.x_val, self.y_val = validation_data
        # 保存每個epoch的預測統計
        self.prediction_stats = []
    
    def on_epoch_end(self, epoch, logs=None):
        # 預測驗證集
        y_pred = self.model.predict(self.x_val, verbose=0)
        
        # 計算統計信息
        pred_min = float(np.min(y_pred))
        pred_max = float(np.max(y_pred))
        pred_mean = float(np.mean(y_pred))
        pred_std = float(np.std(y_pred))
        unique_count = len(np.unique(np.round(y_pred, 3)))
        
        # 儲存統計數據
        stats = {
            'epoch': epoch, 
            'pred_min': pred_min,
            'pred_max': pred_max,
            'pred_mean': pred_mean,
            'pred_std': pred_std,
            'unique_values': unique_count
        }
        self.prediction_stats.append(stats)
        
        # 添加到日誌
        if logs is not None:
            logs['pred_min'] = pred_min
            logs['pred_max'] = pred_max
            logs['pred_mean'] = pred_mean
            logs['pred_std'] = pred_std
            logs['unique_values'] = unique_count
        
        print(f"\n預測統計 - min: {pred_min:.4f}, max: {pred_max:.4f}, mean: {pred_mean:.4f}, std: {pred_std:.4f}, unique: {unique_count}")
        
        # 檢查模型是否學習
        if epoch > 0 and unique_count < 5:
            print(f"警告: 預測值多樣性很低! 模型可能沒有有效學習")

def create_value_model_1():
    """創建簡單的價值預測模型 - 使用CNN架構"""
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
    return model

def create_value_model_2():
    """創建更簡單的價值預測模型 - 使用ResNet風格架構"""
    inputs = tf.keras.layers.Input(shape=(19, 19, 3))
    
    # 初始卷積
    x = tf.keras.layers.Conv2D(32, (3, 3), padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    
    # 殘差塊
    def residual_block(x, filters):
        shortcut = x
        
        x = tf.keras.layers.Conv2D(filters, (3, 3), padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        
        x = tf.keras.layers.Conv2D(filters, (3, 3), padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        
        x = tf.keras.layers.add([shortcut, x])
        x = tf.keras.layers.Activation('relu')(x)
        return x
    
    # 添加殘差塊
    x = residual_block(x, 32)
    x = residual_block(x, 32)
    
    # 全局池化
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    
    # 全連接層
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    outputs = tf.keras.layers.Dense(1, activation='tanh')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

def huber_loss(y_true, y_pred, delta=1.0):
    """Huber損失函數，對異常值更加穩健"""
    error = y_true - y_pred
    abs_error = tf.abs(error)
    quadratic = tf.minimum(abs_error, delta)
    linear = abs_error - quadratic
    return tf.reduce_mean(0.5 * tf.square(quadratic) + delta * linear)

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
    
    # 數據隨機化 - 確保訓練和驗證集是完全隨機的
    indices = np.random.permutation(len(X_board))
    X_board = X_board[indices]
    y_value = y_value[indices]
    
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
    
    # 使用多種模型架構和優化器進行測試
    models_to_test = [
        {"name": "CNN_Adam", "model": create_value_model_1(), "optimizer": tf.keras.optimizers.Adam(learning_rate=0.01)},
        {"name": "CNN_SGD", "model": create_value_model_1(), "optimizer": tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)},
        {"name": "ResNet_Adam", "model": create_value_model_2(), "optimizer": tf.keras.optimizers.Adam(learning_rate=0.01)},
        {"name": "ResNet_SGD", "model": create_value_model_2(), "optimizer": tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)}
    ]
    
    # 開始測試每個模型
    all_results = {}
    best_model = None
    best_mae = float('inf')
    
    for model_config in models_to_test:
        model_name = model_config["name"]
        model = model_config["model"]
        optimizer = model_config["optimizer"]
        
        logger.info(f"\n訓練模型: {model_name}")
        
        # 編譯模型 - 嘗試自定義損失和標準損失
        if model_name.endswith("_SGD"):
            # 對SGD使用Huber損失，提高穩定性
            model.compile(
                optimizer=optimizer,
                loss=huber_loss,  # 使用Huber損失
                metrics=['mae', tf.keras.metrics.RootMeanSquaredError()]
            )
        else:
            # 對Adam使用標準MSE
            model.compile(
                optimizer=optimizer,
                loss='mse',
                metrics=['mae', tf.keras.metrics.RootMeanSquaredError()]
            )
        
        # 設置回調
        prediction_check = PredictionCheckCallback((X_val, y_val))
        
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3),
            # 保存每個模型 - 修正參數名稱
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(output_dir, f'{model_name}_best.keras'),
                save_best_only=True,  # 正確的參數名稱
                monitor='val_mae',
                mode='min'
            ),
            # 添加預測檢查回調
            prediction_check
        ]
        
        # 訓練模型
        logger.info(f"開始訓練 {model_name}...")
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=20,
            batch_size=64,
            callbacks=callbacks,
            verbose=1
        )
        
        # 評估模型
        results = model.evaluate(X_val, y_val, verbose=1)
        metrics = model.metrics_names
        
        # 記錄結果
        eval_results = {}
        for name, value in zip(metrics, results):
            eval_results[name] = value
            logger.info(f"{model_name} - {name}: {value:.4f}")
        
        # 檢查是否是最佳模型
        current_mae = eval_results.get('mae', float('inf'))
        if current_mae < best_mae:
            best_mae = current_mae
            best_model = model
            logger.info(f"新的最佳模型: {model_name}，MAE: {best_mae:.4f}")
        
        # 保存預測檢查結果
        all_results[model_name] = {
            "history": history.history,
            "evaluation": eval_results,
            "prediction_stats": prediction_check.prediction_stats
        }
        
        # 預測並繪製散點圖
        y_pred = model.predict(X_val)
        
        plt.figure(figsize=(10, 10))
        plt.scatter(y_val, y_pred, alpha=0.5)
        plt.plot([-1, 1], [-1, 1], 'r--')
        plt.xlabel('真實價值')
        plt.ylabel('預測價值')
        plt.title(f'{model_name} 價值預測散點圖')
        plt.grid(True)
        plt.xlim([-1.1, 1.1])
        plt.ylim([-1.1, 1.1])
        plt.savefig(os.path.join(output_dir, f'{model_name}_predictions.png'))
        plt.close()
        
        # 繪製訓練歷史
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title(f'{model_name} Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(2, 2, 2)
        plt.plot(history.history['mae'], label='Training MAE')
        plt.plot(history.history['val_mae'], label='Validation MAE')
        plt.title(f'{model_name} MAE')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()
        
        # 繪製預測統計變化
        plt.subplot(2, 2, 3)
        epochs = [stat['epoch'] for stat in prediction_check.prediction_stats]
        min_vals = [stat['pred_min'] for stat in prediction_check.prediction_stats]
        max_vals = [stat['pred_max'] for stat in prediction_check.prediction_stats]
        mean_vals = [stat['pred_mean'] for stat in prediction_check.prediction_stats]
        std_vals = [stat['pred_std'] for stat in prediction_check.prediction_stats]
        
        plt.plot(epochs, min_vals, label='Min Pred')
        plt.plot(epochs, max_vals, label='Max Pred')
        plt.plot(epochs, mean_vals, label='Mean Pred')
        plt.plot(epochs, std_vals, label='Std Pred')
        plt.title(f'{model_name} Prediction Stats')
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.legend()
        
        # 繪製預測唯一值數量
        plt.subplot(2, 2, 4)
        unique_vals = [stat['unique_values'] for stat in prediction_check.prediction_stats]
        plt.plot(epochs, unique_vals, 'o-')
        plt.title(f'{model_name} Unique Predictions')
        plt.xlabel('Epoch')
        plt.ylabel('Count')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{model_name}_history.png'))
        plt.close()
        
        # 釋放記憶體
        tf.keras.backend.clear_session()
        time.sleep(2)  # 給系統時間清理記憶體
    
    # 比較所有模型的性能
    logger.info("\n所有模型性能比較:")
    mae_values = []
    model_names = []
    
    for model_name, results in all_results.items():
        mae = results["evaluation"].get("mae", float('inf'))
        logger.info(f"{model_name}: MAE = {mae:.4f}")
        mae_values.append(mae)
        model_names.append(model_name)
    
    # 繪製所有模型的MAE比較
    plt.figure(figsize=(12, 6))
    plt.bar(model_names, mae_values)
    plt.title('All Models MAE Comparison')
    plt.xlabel('Model')
    plt.ylabel('MAE (lower is better)')
    for i, v in enumerate(mae_values):
        plt.text(i, v + 0.01, f"{v:.4f}", ha='center')
    plt.savefig(os.path.join(output_dir, 'model_comparison.png'))
    plt.close()
    
    # 實現自定義訓練循環
    logger.info("\n嘗試自定義訓練循環...")
    best_model_custom = create_value_model_2()  # 使用ResNet風格模型
    
    # 自定義訓練循環
    def custom_train(model, x_train, y_train, x_val, y_val, epochs=10, batch_size=64):
        optimizer = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)
        loss_fn = tf.keras.losses.MeanSquaredError()
        mae_metric = tf.keras.metrics.MeanAbsoluteError()
        
        # 準備數據集
        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
        
        # 訓練歷史記錄
        history = {
            'loss': [],
            'val_loss': [],
            'mae': [],
            'val_mae': []
        }
        
        # 跟踪預測變化
        prediction_stats = []
        
        # 訓練循環
        for epoch in range(epochs):
            logger.info(f"Epoch {epoch+1}/{epochs}")
            
            # 重置指標
            train_loss = tf.keras.metrics.Mean()
            train_mae = tf.keras.metrics.Mean()
            
            # 訓練步驟
            for step, (x_batch, y_batch) in enumerate(train_dataset):
                with tf.GradientTape() as tape:
                    predictions = model(x_batch, training=True)
                    loss = loss_fn(y_batch, predictions)
                
                gradients = tape.gradient(loss, model.trainable_variables)
                
                # 輸出梯度信息（前幾個批次）
                if step < 3:
                    grad_norm = tf.linalg.global_norm(gradients)
                    print(f"  Batch {step+1} - 梯度範數: {grad_norm.numpy():.4f}")
                
                # 應用梯度裁剪
                gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                
                # 更新指標
                train_loss.update_state(loss)
                mae_metric.reset_states()
                mae_metric.update_state(y_batch, predictions)
                train_mae.update_state(mae_metric.result())
                
                if step % 100 == 0:
                    print(f"  Batch {step+1} - Loss: {train_loss.result().numpy():.4f}, MAE: {train_mae.result().numpy():.4f}")
            
            # 計算驗證集指標
            val_predictions = model.predict(x_val, verbose=0)
            val_loss = loss_fn(y_val, val_predictions).numpy()
            mae_metric.reset_states()
            mae_metric.update_state(y_val, val_predictions)
            val_mae = mae_metric.result().numpy()
            
            # 記錄訓練歷史
            history['loss'].append(train_loss.result().numpy())
            history['val_loss'].append(val_loss)
            history['mae'].append(train_mae.result().numpy())
            history['val_mae'].append(val_mae)
            
            # 檢查預測統計
            pred_min = np.min(val_predictions)
            pred_max = np.max(val_predictions)
            pred_mean = np.mean(val_predictions)
            pred_std = np.std(val_predictions)
            unique_count = len(np.unique(np.round(val_predictions, 3)))
            
            prediction_stats.append({
                'epoch': epoch,
                'pred_min': float(pred_min),
                'pred_max': float(pred_max),
                'pred_mean': float(pred_mean),
                'pred_std': float(pred_std),
                'unique_values': unique_count
            })
            
            logger.info(f"Epoch {epoch+1} - Loss: {train_loss.result():.4f}, MAE: {train_mae.result():.4f}, Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.4f}")
            logger.info(f"預測統計 - min: {pred_min:.4f}, max: {pred_max:.4f}, mean: {pred_mean:.4f}, std: {pred_std:.4f}, unique: {unique_count}")
        
        return history, prediction_stats
    
    # 使用自定義訓練循環
    custom_history, custom_pred_stats = custom_train(
        best_model_custom, 
        X_train, y_train, 
        X_val, y_val, 
        epochs=10, 
        batch_size=64
    )
    
    # 評估自定義訓練的模型
    y_pred_custom = best_model_custom.predict(X_val)
    mae_custom = np.mean(np.abs(y_val - y_pred_custom))
    mse_custom = np.mean(np.square(y_val - y_pred_custom))
    logger.info(f"自定義訓練 - MAE: {mae_custom:.4f}, MSE: {mse_custom:.4f}")
    
    # 繪製自定義訓練結果
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(custom_history['loss'], label='Training Loss')
    plt.plot(custom_history['val_loss'], label='Validation Loss')
    plt.title('Custom Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(2, 2, 2)
    plt.plot(custom_history['mae'], label='Training MAE')
    plt.plot(custom_history['val_mae'], label='Validation MAE')
    plt.title('Custom Training MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    
    # 繪製預測統計變化
    plt.subplot(2, 2, 3)
    epochs = [stat['epoch'] for stat in custom_pred_stats]
    min_vals = [stat['pred_min'] for stat in custom_pred_stats]
    max_vals = [stat['pred_max'] for stat in custom_pred_stats]
    mean_vals = [stat['pred_mean'] for stat in custom_pred_stats]
    std_vals = [stat['pred_std'] for stat in custom_pred_stats]
    
    plt.plot(epochs, min_vals, label='Min Pred')
    plt.plot(epochs, max_vals, label='Max Pred')
    plt.plot(epochs, mean_vals, label='Mean Pred')
    plt.plot(epochs, std_vals, label='Std Pred')
    plt.title('Custom Training Prediction Stats')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()
    
    # 繪製預測唯一值數量
    plt.subplot(2, 2, 4)
    unique_vals = [stat['unique_values'] for stat in custom_pred_stats]
    plt.plot(epochs, unique_vals, 'o-')
    plt.title('Custom Training Unique Predictions')
    plt.xlabel('Epoch')
    plt.ylabel('Count')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'custom_training_history.png'))
    plt.close()
    
    # 預測並繪製散點圖
    plt.figure(figsize=(10, 10))
    plt.scatter(y_val, y_pred_custom, alpha=0.5)
    plt.plot([-1, 1], [-1, 1], 'r--')
    plt.xlabel('真實價值')
    plt.ylabel('預測價值')
    plt.title('Custom Training 價值預測散點圖')
    plt.grid(True)
    plt.xlim([-1.1, 1.1])
    plt.ylim([-1.1, 1.1])
    plt.savefig(os.path.join(output_dir, 'custom_training_predictions.png'))
    plt.close()
    
    logger.info("診斷完成，結果已保存到 " + output_dir)
    
    # 返回最佳模型和所有結果
    return best_model, all_results

# 運行診斷
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        data_file = sys.argv[1]
    else:
        data_file = "process_kgs_dataset_50000.npz"  # 預設數據文件
    
    # 使用2%的數據進行測試 - 減小樣本量加快測試
    diagnose_value_prediction(data_file, sample_rate=0.02)