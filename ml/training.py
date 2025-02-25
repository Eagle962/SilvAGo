import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import time
from tqdm import tqdm
from game.go_game import GoGame
from search.mcts import MCTS
from data.dataset import create_tf_dataset
from datetime import datetime
import json
import logging

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger(__name__)

class TrainingPipeline:
    """圍棋AI訓練管線。"""
    
    def __init__(self, model, config=None):
        """初始化訓練管線。
        
        Args:
            model: 神經網絡模型
            config: 訓練配置字典
        """
        self.model = model
        
        # 默認配置
        self.config = {
            'batch_size': 256,
            'epochs_per_iteration': 10,
            'validation_split': 0.1,
            'num_self_play_games': 400,
            'num_mcts_simulations': 800,
            'temperature': 1.0,
            'dirichlet_epsilon': 0.25,
            'dirichlet_alpha': 0.03,
            'c_puct': 5.0,
            'memory_size': 500000,
            'min_buffer_size': 10000,
            'min_train_iterations': 5,
            'lr_schedule': [(0, 0.01), (100000, 0.001), (300000, 0.0001)],
            'momentum': 0.9,
            'weight_decay': 1e-4,
            'checkpoint_freq': 1,
            'output_dir': 'models',
            'log_dir': 'logs',
            'use_mixed_precision': False,
            'distribution_strategy': 'mirrored',
            'early_stopping_patience': 5
        }
        
        # 更新配置
        if config is not None:
            self.config.update(config)
        
        # 建立輸出目錄
        os.makedirs(self.config['output_dir'], exist_ok=True)
        os.makedirs(self.config['log_dir'], exist_ok=True)
        
        # 初始化訓練步數
        self.train_step = 0
        
        # 經驗回放緩衝區
        self.replay_buffer = {
            'states': [],
            'policies': [],
            'values': []
        }
        
        # 初始化分佈策略
        if self.config['distribution_strategy'] == 'mirrored':
            self.strategy = tf.distribute.MirroredStrategy()
        elif self.config['distribution_strategy'] == 'tpu':
            # 如果有TPU可用
            try:
                resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
                tf.config.experimental_connect_to_cluster(resolver)
                tf.tpu.experimental.initialize_tpu_system(resolver)
                self.strategy = tf.distribute.TPUStrategy(resolver)
                logger.info("使用TPU訓練")
            except ValueError:
                logger.warning("無法連接到TPU，使用MirroredStrategy替代")
                self.strategy = tf.distribute.MirroredStrategy()
        else:
            # 默認，單GPU或CPU
            self.strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
        
        # 設置混合精度
        if self.config['use_mixed_precision']:
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
            logger.info("啟用混合精度訓練")
        
        # 使用策略範圍重新編譯模型
        with self.strategy.scope():
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=self.get_lr()),
                loss={
                    'policy': 'categorical_crossentropy',
                    'value': 'mean_squared_error'
                },
                metrics={
                    'policy': 'accuracy',
                    'value': 'mean_absolute_error'
                }
            )
        
        # 保存配置
        self.save_config()
        
        # TensorBoard回調
        self.tb_callback = tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(self.config['log_dir'], 'tensorboard'),
            update_freq='epoch',
            profile_batch=0
        )
    
    def save_config(self):
        """保存訓練配置。"""
        config_path = os.path.join(self.config['output_dir'], 'config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=4)
        logger.info(f"配置已保存到 {config_path}")
    
    def get_lr(self):
        """根據lr_schedule獲取當前學習率。"""
        for step, lr in reversed(self.config['lr_schedule']):
            if self.train_step >= step:
                return lr
        return self.config['lr_schedule'][0][1]
    
    def add_to_replay_buffer(self, states, policies, values):
        """向經驗回放緩衝區添加數據。"""
        # 添加新數據
        self.replay_buffer['states'].extend(states)
        self.replay_buffer['policies'].extend(policies)
        self.replay_buffer['values'].extend(values)
        
        # 限制大小
        max_size = self.config['memory_size']
        if len(self.replay_buffer['states']) > max_size:
            excess = len(self.replay_buffer['states']) - max_size
            self.replay_buffer['states'] = self.replay_buffer['states'][excess:]
            self.replay_buffer['policies'] = self.replay_buffer['policies'][excess:]
            self.replay_buffer['values'] = self.replay_buffer['values'][excess:]
    
    def get_replay_buffer_data(self):
        """從經驗回放緩衝區獲取訓練數據。"""
        return {
            'X_board': np.array(self.replay_buffer['states']),
            'y_policy': np.array(self.replay_buffer['policies']),
            'y_value': np.array(self.replay_buffer['values'])
        }
    
    def self_play(self, num_games=None):
        """生成自我對弈資料。
        
        Args:
            num_games: 自我對弈遊戲數量，默認使用配置中的值
            
        Returns:
            tuple: (states, policies, values)
        """
        if num_games is None:
            num_games = self.config['num_self_play_games']
        
        # 創建MCTS搜索器
        mcts = MCTS(
            model=self.model,
            num_simulations=self.config['num_mcts_simulations'],
            c_puct=self.config['c_puct'],
            dirichlet_epsilon=self.config['dirichlet_epsilon'],
            dirichlet_alpha=self.config['dirichlet_alpha'],
            temperature=self.config['temperature']
        )
        
        all_states = []
        all_policies = []
        all_values = []
        
        # 遊戲計數
        for game_idx in tqdm(range(num_games), desc="自我對弈"):
            game = GoGame()
            game_states = []
            game_policies = []
            current_players = []
            
            # 一盤遊戲進行直到結束
            while not game.is_game_over():
                # 根據遊戲進行階段調整溫度
                progress = len(game.history) / 200.0  # 假設遊戲大約持續200步
                temp = 1.0 if progress < 0.3 else 0.5 if progress < 0.75 else 0.25
                
                # 使用MCTS搜索獲取移動概率分佈
                probs, _ = mcts.get_action_probs(game, temperature=temp)
                
                # 記錄當前狀態
                state_features = game.get_state_features()
                game_states.append(state_features)
                
                # 轉換移動概率為神經網絡輸出格式
                policy = np.zeros(game.size * game.size, dtype=np.float32)
                for move, prob in probs.items():
                    if move != 'pass':
                        x, y = move
                        policy[x * game.size + y] = prob
                
                game_policies.append(policy)
                current_players.append(game.current_player)
                
                # 根據概率選擇移動
                moves = list(probs.keys())
                probs_list = [probs[move] for move in moves]
                selected_move = moves[np.random.choice(len(moves), p=probs_list)]
                
                # 執行移動
                if not game.play(selected_move):
                    # 如果移動失敗（非法），則虛手
                    game.play('pass')
            
            # 獲取遊戲結果
            result = game.get_winner()
            winner = result[0] if result else 0
            
            # 轉換價值以匹配AlphaZero格式：從當前玩家的角度看，1=贏，-1=輸
            values = [winner * player for player in current_players]
            
            # 添加到總數據
            all_states.extend(game_states)
            all_policies.extend(game_policies)
            all_values.extend(values)
            
            # 每10場遊戲打印一次狀態
            if (game_idx + 1) % 10 == 0:
                avg_value = np.mean(values) if values else 0
                logger.info(f"完成 {game_idx + 1} 場遊戲，平均價值: {avg_value:.3f}")
        
        return np.array(all_states), np.array(all_policies), np.array(all_values)
    
    def train_network(self, data=None, epochs=None, initial_epoch=0):
        """訓練神經網絡。
        
        Args:
            data: 訓練數據字典，含'X_board', 'y_policy', 'y_value'
            epochs: 訓練輪數
            initial_epoch: 起始輪數
            
        Returns:
            訓練歷史
        """
        # 使用配置中的默認值
        if epochs is None:
            epochs = self.config['epochs_per_iteration']
        
        # 如果沒有提供數據，使用經驗回放緩衝區
        if data is None:
            data = self.get_replay_buffer_data()
        
        # 確保數據非空
        if len(data['X_board']) == 0:
            logger.warning("沒有訓練數據，跳過訓練")
            return None
        
        logger.info(f"開始訓練，數據形狀: {data['X_board'].shape}")
        
        # 創建數據集
        train_size = int(len(data['X_board']) * (1 - self.config['validation_split']))
        
        # 訓練集
        train_data = {
            'X_board': data['X_board'][:train_size],
            'y_policy': data['y_policy'][:train_size],
            'y_value': data['y_value'][:train_size]
        }
        
        # 驗證集
        val_data = {
            'X_board': data['X_board'][train_size:],
            'y_policy': data['y_policy'][train_size:],
            'y_value': data['y_value'][train_size:]
        }
        
        # 創建TensorFlow數據集
        train_dataset = create_tf_dataset(
            train_data, 
            batch_size=self.config['batch_size'],
            shuffle_buffer=10000,
            repeat=True
        )
        
        val_dataset = create_tf_dataset(
            val_data,
            batch_size=self.config['batch_size'],
            shuffle_buffer=0,
            repeat=False
        )
        
        # 設置當前學習率
        K = tf.keras.backend
        lr = self.get_lr()
        K.set_value(self.model.optimizer.learning_rate, lr)
        logger.info(f"當前學習率: {lr}")
        
        # 每個訓練輪次的步數
        steps_per_epoch = max(1, train_size // self.config['batch_size'])
        validation_steps = max(1, (len(data['X_board']) - train_size) // self.config['batch_size'])
        
        # 設置回調
        callbacks = [
            self.tb_callback,
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(self.config['output_dir'], 'checkpoint_{epoch:02d}'),
                save_best_only=True,
                monitor='val_loss',
                verbose=1
            ),
            tf.keras.callbacks.EarlyStopping(
                patience=self.config['early_stopping_patience'],
                restore_best_weights=True,
                monitor='val_loss'
            ),
            tf.keras.callbacks.LearningRateScheduler(lambda epoch, lr: self.get_lr())
        ]
        
        # 訓練模型
        history = self.model.fit(
            train_dataset,
            epochs=initial_epoch + epochs,
            initial_epoch=initial_epoch,
            steps_per_epoch=steps_per_epoch,
            validation_data=val_dataset,
            validation_steps=validation_steps,
            callbacks=callbacks,
            verbose=1
        )
        
        # 更新訓練步數
        self.train_step += steps_per_epoch * epochs
        
        # 保存模型
        model_path = os.path.join(
            self.config['output_dir'],
            f'model_step_{self.train_step}'
        )
        self.model.save(model_path)
        logger.info(f"模型已保存到 {model_path}")
        
        # 繪製訓練歷史
        self.plot_training_history(history)
        
        return history
    
    def plot_training_history(self, history):
        """繪製訓練歷史曲線。"""
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
        plt.plot(history.history['policy_accuracy'], label='Training Policy Accuracy')
        plt.plot(history.history['val_policy_accuracy'], label='Validation Policy Accuracy')
        plt.title('Policy Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # 價值MAE
        plt.subplot(1, 3, 3)
        plt.plot(history.history['value_mean_absolute_error'], label='Training Value MAE')
        plt.plot(history.history['val_value_mean_absolute_error'], label='Validation Value MAE')
        plt.title('Value Mean Absolute Error')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()
        
        plt.tight_layout()
        
        # 保存圖片
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        plt.savefig(os.path.join(self.config['log_dir'], f'training_history_{timestamp}.png'))
        plt.close()
    
    def evaluate_model(self, opponent_model=None, num_games=100):
        """評估模型性能。
        
        Args:
            opponent_model: 對手模型，如果為None，則使用隨機走子
            num_games: 評估遊戲數量
            
        Returns:
            dict: 評估結果
        """
        # 準備評估
        current_mcts = MCTS(
            model=self.model,
            num_simulations=self.config['num_mcts_simulations'],
            temperature=0.1  # 低溫度，幾乎確定性選擇最佳移動
        )
        
        # 對手搜索器
        if opponent_model is not None:
            opponent_mcts = MCTS(
                model=opponent_model,
                num_simulations=self.config['num_mcts_simulations'],
                temperature=0.1
            )
        
        # 結果統計
        results = {
            'wins': 0,
            'losses': 0,
            'draws': 0,
            'avg_game_length': 0,
            'total_moves': 0
        }
        
        # 進行評估遊戲
        for game_idx in tqdm(range(num_games), desc="評估"):
            game = GoGame()
            move_count = 0
            
            # 確定當前模型為先手（黑）還是後手（白）
            current_model_color = 1 if game_idx % 2 == 0 else -1
            
            # 一盤遊戲進行直到結束
            while not game.is_game_over():
                # 確定當前行動的模型
                is_current_model_turn = (game.current_player == current_model_color)
                
                if is_current_model_turn:
                    # 使用當前模型
                    probs, _ = current_mcts.get_action_probs(game, temperature=0.1)
                else:
                    # 使用對手模型或隨機走子
                    if opponent_model is not None:
                        probs, _ = opponent_mcts.get_action_probs(game, temperature=0.1)
                    else:
                        # 隨機走子
                        legal_moves = game.get_legal_moves()
                        probs = {move: 1.0 / len(legal_moves) for move in legal_moves}
                
                # 選擇最佳移動
                best_move = max(probs.items(), key=lambda x: x[1])[0]
                
                # 應用移動
                success = game.play(best_move)
                if not success:
                    # 如果移動失敗，執行虛手
                    game.play('pass')
                
                move_count += 1
            
            # 獲取結果
            result = game.get_winner()
            winner = result[0] if result else 0
            
            # 更新統計
            if winner == 0:
                results['draws'] += 1
            elif (winner == 1 and current_model_color == 1) or (winner == -1 and current_model_color == -1):
                results['wins'] += 1
            else:
                results['losses'] += 1
            
            results['total_moves'] += move_count
        
        # 計算平均遊戲長度和勝率
        results['avg_game_length'] = results['total_moves'] / num_games
        results['win_rate'] = results['wins'] / num_games
        
        # 記錄結果
        logger.info(f"評估結果: 勝: {results['wins']}, 負: {results['losses']}, 平: {results['draws']}")
        logger.info(f"勝率: {results['win_rate']:.2f}, 平均遊戲長度: {results['avg_game_length']:.1f}")
        
        # 保存結果
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        results_path = os.path.join(self.config['log_dir'], f'evaluation_{timestamp}.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)
        
        return results
    
    def training_iteration(self, initial_data=None):
        """執行一個完整的訓練迭代。
        
        包含自我對弈、神經網絡訓練和模型評估。
        
        Args:
            initial_data: 初始訓練數據
            
        Returns:
            訓練歷史
        """
        # 1. 生成自我對弈資料
        logger.info("開始自我對弈...")
        states, policies, values = self.self_play()
        
        # 2. 添加到經驗回放緩衝區
        self.add_to_replay_buffer(states, policies, values)
        
        # 如果有初始數據，也添加到緩衝區
        if initial_data is not None:
            self.add_to_replay_buffer(
                initial_data['X_board'],
                initial_data['y_policy'],
                initial_data['y_value']
            )
        
        # 檢查緩衝區大小
        buffer_size = len(self.replay_buffer['states'])
        logger.info(f"經驗回放緩衝區大小: {buffer_size}")
        
        # 緩衝區不夠大，跳過訓練
        if buffer_size < self.config['min_buffer_size'] and buffer_size < self.config['min_train_iterations']:
            logger.info(f"緩衝區不足 {self.config['min_buffer_size']} 個樣本，跳過訓練")
            return None
        
        # 3. 訓練神經網絡
        logger.info("開始訓練神經網絡...")
        history = self.train_network()
        
        # 4. 評估模型
        if buffer_size >= self.config['min_buffer_size']:
            logger.info("評估模型...")
            self.evaluate_model()
        
        return history
    
    def run_training(self, num_iterations, initial_data=None):
        """運行完整的訓練循環。
        
        Args:
            num_iterations: 訓練迭代次數
            initial_data: 初始訓練數據
        """
        logger.info(f"開始訓練循環，計劃執行 {num_iterations} 次迭代")
        
        for i in range(num_iterations):
            logger.info(f"===== 迭代 {i+1}/{num_iterations} =====")
            start_time = time.time()
            
            # 執行一次完整訓練迭代
            history = self.training_iteration(initial_data if i == 0 else None)
            
            # 記錄本次迭代時間
            elapsed = time.time() - start_time
            logger.info(f"迭代 {i+1} 完成，耗時: {elapsed:.1f} 秒")
            
            # 每次迭代後保存完整模型
            model_path = os.path.join(
                self.config['output_dir'],
                f'model_iteration_{i+1}'
            )
            self.model.save(model_path)
            logger.info(f"已保存迭代 {i+1} 的模型到 {model_path}")
        
        logger.info("訓練循環完成")