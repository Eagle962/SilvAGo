import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import heapq
from tqdm import tqdm
import matplotlib.pyplot as plt

# 使用與原始代碼相同的GoGame類
class GoGame:
    def __init__(self):
        self.size = 19
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.current_player = 1  # 1 代表黑棋, -1 代表白棋
        self.ko = None
        self.last_move = None
        self.passes = 0
        self.captured_stones = {1: 0, -1: 0}  # 黑棋: 0, 白棋: 0
        self.komi = 6.5  # 貼目
        self.history = [] 
        self.dead_stones = set()  # 用於標記掛掉的棋子

    def play(self, move):
        if move == 'pass':
            self.passes += 1
            self.current_player = -self.current_player
            self.history.append(self.board.copy())
            return True

        if not self.is_valid_move(move):
            return False

        x, y = move
        self.board[x, y] = self.current_player
        captured = self.remove_captured_stones(self.get_opponent(self.current_player))
        
        if self.is_suicide(x, y) and not captured:
            self.board[x, y] = 0
            return False

        self.remove_captured_stones(self.current_player)
        self.last_move = move
        self.passes = 0
        self.current_player = self.get_opponent(self.current_player)
        
        # 檢查避免出現重複動作
        board_state = self.board.copy()
        if any(np.array_equal(board_state, hist) for hist in self.history):
            self.board[x, y] = 0
            return False
        
        self.history.append(board_state)
        return True

    def is_valid_move(self, move):
        if move == 'pass':
            return True
        x, y = move
        if x < 0 or x >= self.size or y < 0 or y >= self.size:
            return False
        if self.board[x, y] != 0:
            return False
        return True

    def get_opponent(self, player):
        return -player

    def get_adjacent_points(self, x, y):
        adjacent = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.size and 0 <= ny < self.size:
                adjacent.append((nx, ny))
        return adjacent

    def get_group(self, x, y):
        color = self.board[x, y]
        group = set([(x, y)])
        frontier = [(x, y)]
        while frontier:
            current = frontier.pop()
            for adj in self.get_adjacent_points(*current):
                if self.board[adj] == color and adj not in group:
                    group.add(adj)
                    frontier.append(adj)
        return group

    def get_liberties(self, group):
        liberties = set()
        for x, y in group:
            for adj in self.get_adjacent_points(x, y):
                if self.board[adj[0], adj[1]] == 0:
                    liberties.add(adj)
        return liberties

    def remove_captured_stones(self, player):
        captured = set()
        for x in range(self.size):
            for y in range(self.size):
                if self.board[x, y] == player:
                    group = self.get_group(x, y)
                    if not self.get_liberties(group):
                        for stone in group:
                            self.board[stone] = 0
                            captured.add(stone)
        self.captured_stones[self.get_opponent(player)] += len(captured)
        return captured

    def is_suicide(self, x, y):
        group = self.get_group(x, y)
        return len(self.get_liberties(group)) == 0

    def get_legal_moves(self):
        moves = ['pass']
        for x in range(self.size):
            for y in range(self.size):
                if self.is_valid_move((x, y)):
                    moves.append((x, y))
        return moves

    def is_game_over(self):
        return self.passes >= 2

    def mark_dead_stones(self):
        pass

    def get_winner(self):
        if not self.is_game_over():
            return None
        
        self.mark_dead_stones()  # 在計算得分前標記掛掉的棋子
        black_score, white_score = self.calculate_score()
        
        if black_score > white_score:
            return 1, black_score, white_score
        elif white_score > black_score:
            return -1, black_score, white_score
        else:
            return 0, black_score, white_score  # 平局

    def calculate_score(self):
        territory = self.calculate_territory()
        black_score = np.sum(self.board == 1) + self.captured_stones[1] + np.sum(territory == 1)
        white_score = np.sum(self.board == -1) + self.captured_stones[-1] + np.sum(territory == -1) + self.komi
        
        # 把死掉的棋子移掉後計算分數
        for x, y in self.dead_stones:
            if self.board[x, y] == 1:
                black_score -= 1
                white_score += 1
            elif self.board[x, y] == -1:
                white_score -= 1
                black_score += 1
        
        return black_score, white_score

    def calculate_territory(self):
        territory = np.zeros((self.size, self.size))
        visited = set()

        def flood_fill(x, y, color):
            if (x, y) in visited or not (0 <= x < self.size and 0 <= y < self.size):
                return set(), set(), color
            visited.add((x, y))
            if self.board[x, y] != 0 and (x, y) not in self.dead_stones:
                return set(), set(), self.board[x, y]
            
            empty = {(x, y)}
            border = set()
            for nx, ny in self.get_adjacent_points(x, y):
                n_empty, n_border, n_color = flood_fill(nx, ny, color)
                empty |= n_empty
                border |= n_border
                if n_color != 0 and color == 0:
                    color = n_color
                elif n_color != 0 and n_color != color:
                    color = 0  # 無主的領地
            if color == 0:
                border |= {(x, y)}
            return empty, border, color

        for x in range(self.size):
            for y in range(self.size):
                if (x, y) not in visited and (self.board[x, y] == 0 or (x, y) in self.dead_stones):
                    empty, border, color = flood_fill(x, y, 0)
                    if color != 0:
                        for ex, ey in empty:
                            territory[ex, ey] = color

        return territory

# CNN-RNN混合模型定義
def create_cnn_rnn_model(input_shape=(19, 19, 3), sequence_length=8):
    # CNN部分
    cnn_input = layers.Input(shape=input_shape)
    
    # 初始卷積層
    x = layers.Conv2D(64, (3, 3), padding='same')(cnn_input)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    # 残差块
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
    
    # 添加多個殘差塊
    for _ in range(5):
        x = residual_block(x, 64)
    
    # 特徵提取
    cnn_features = layers.Conv2D(128, (3, 3), padding='same')(x)
    cnn_features = layers.BatchNormalization()(cnn_features)
    cnn_features = layers.ReLU()(cnn_features)
    
    # 展平CNN特徵
    cnn_flat = layers.Reshape((19*19, 128))(cnn_features)
    
    # RNN部分
    rnn_input = layers.Input(shape=(sequence_length, 19*19*3))
    rnn = layers.LSTM(256, return_sequences=True)(rnn_input)
    rnn = layers.LSTM(256)(rnn)
    
    # 合併CNN和RNN特徵
    combined = layers.Concatenate()([layers.Flatten()(cnn_features), rnn])
    
    # 策略頭
    policy = layers.Dense(512, activation='relu')(combined)
    policy = layers.Dropout(0.3)(policy)
    policy = layers.Dense(361, activation='softmax', name='policy')(policy)
    
    # 價值頭
    value = layers.Dense(256, activation='relu')(combined)
    value = layers.Dropout(0.3)(value)
    value = layers.Dense(1, activation='tanh', name='value')(value)
    
    model = models.Model(inputs=[cnn_input, rnn_input], 
                        outputs=[policy, value])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss={'policy': 'categorical_crossentropy', 'value': 'mse'},
        metrics={'policy': 'accuracy', 'value': 'mae'}
    )
    
    return model

# A*搜索節點類
class AStarNode:
    def __init__(self, game_state, parent=None, move=None, g_cost=0, h_cost=0):
        self.game_state = game_state
        self.parent = parent
        self.move = move
        self.g_cost = g_cost  # 實際成本
        self.h_cost = h_cost  # 啟發式估計成本
        self.f_cost = g_cost + h_cost
        
    def __lt__(self, other):
        return self.f_cost < other.f_cost

# A*搜索算法
class AStarSearch:
    def __init__(self, model, max_depth=10):
        self.model = model
        self.max_depth = max_depth
    
    def get_board_features(self, game):
        features = np.zeros((19, 19, 3), dtype=np.float32)
        features[:,:,0] = (game.board == game.current_player)
        features[:,:,1] = (game.board == -game.current_player)
        features[:,:,2] = game.current_player
        return features
    
    def get_sequence_features(self, game, sequence_length=8):
        sequence = []
        history = list(reversed(game.history[-sequence_length:]))
        current_board = game.board.copy()
        
        # 填充序列至指定長度
        while len(history) < sequence_length:
            history.append(np.zeros_like(current_board))
            
        for board in history[-sequence_length:]:
            features = np.zeros((19*19*3,), dtype=np.float32)
            features[:19*19] = (board == 1).flatten()
            features[19*19:2*19*19] = (board == -1).flatten()
            features[2*19*19:] = game.current_player
            sequence.append(features)
            
        return np.array([sequence])
    
    def evaluate_position(self, game):
        board_features = np.expand_dims(self.get_board_features(game), 0)
        sequence_features = self.get_sequence_features(game)
        
        policy, value = self.model.predict([board_features, sequence_features], verbose=0)
        return policy[0], value[0][0]
    
    def search(self, game):
        start_node = AStarNode(game)
        open_list = [start_node]
        closed_set = set()
        
        while open_list and len(closed_set) < self.max_depth:
            current = heapq.heappop(open_list)
            
            # 終止條件
            if current.game_state.is_game_over():
                return self.backtrack_best_move(current)
            
            state_hash = hash(str(current.game_state.board.tobytes()))
            if state_hash in closed_set:
                continue
                
            closed_set.add(state_hash)
            
            # 生成合法移動
            legal_moves = current.game_state.get_legal_moves()
            policy, value = self.evaluate_position(current.game_state)
            
            for move in legal_moves:
                if move == 'pass':
                    continue
                    
                new_game = GoGame()
                new_game.board = current.game_state.board.copy()
                new_game.current_player = current.game_state.current_player
                new_game.history = current.game_state.history.copy()
                
                if new_game.play(move):
                    move_idx = move[0] * 19 + move[1]
                    g_cost = current.g_cost + 1
                    h_cost = (1 - policy[move_idx]) * (1 - value) * 10
                    
                    new_node = AStarNode(
                        new_game, 
                        parent=current,
                        move=move,
                        g_cost=g_cost,
                        h_cost=h_cost
                    )
                    
                    heapq.heappush(open_list, new_node)
        
        # 如果沒有找到終止狀態，返回最佳評估的移動
        if open_list:
            return self.backtrack_best_move(min(open_list, key=lambda x: x.f_cost))
        return (0, 0)  # 默認移動
    
    def backtrack_best_move(self, node):
        while node.parent and node.parent.parent:
            node = node.parent
        return node.move

# 數據預處理函數
def preprocess_data(game_data, sequence_length=8):
    X_board = []
    X_sequence = []
    y_policy = []
    y_value = []
    
    for game_record in game_data:
        board_state = game_record['board_state']
        move_history = game_record['move_history']
        winner = game_record['winner']
        
        # 處理板面特徵
        X_board.append(board_state)
        
        # 處理序列特徵
        sequence = []
        history = list(reversed(move_history[-sequence_length:]))
        while len(history) < sequence_length:
            history.append(np.zeros_like(board_state))
        sequence = np.array([board.flatten() for board in history])
        X_sequence.append(sequence)
        
        # 處理標籤
        move = game_record['next_move']
        policy = np.zeros(361)
        if move != 'pass':
            policy[move[0] * 19 + move[1]] = 1
        y_policy.append(policy)
        y_value.append(winner)
    
    return np.array(X_board), np.array(X_sequence), np.array(y_policy), np.array(y_value)

# 訓練函數
def train_model(model, train_data, val_data, epochs=30, batch_size=32):
    X_train_board, X_train_seq, y_train_policy, y_train_value = train_data
    X_val_board, X_val_seq, y_val_policy, y_val_value = val_data
    
    history = model.fit(
        [X_train_board, X_train_seq],
        {'policy': y_train_policy, 'value': y_train_value},
        validation_data=(
            [X_val_board, X_val_seq],
            {'policy': y_val_policy, 'value': y_val_value}
        ),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3),
            tf.keras.callbacks.ModelCheckpoint(
                'best_cnn_rnn_model.h5',
                save_best_only=True
            )
        ]
    )
    return history

# 主訓練循環
def create_dataset(data, batch_size):
    X_board, X_seq, y_policy, y_value = data
    dataset = tf.data.Dataset.from_tensor_slices((
        {
            'board_input': X_board,
            'sequence_input': X_seq
        },
        {
            'policy': y_policy,
            'value': y_value
        }
    ))
    return dataset.shuffle(10000).batch(batch_size).prefetch(tf.data.AUTOTUNE)

def train_model(model, train_data, val_data, epochs=30, batch_size=2048):
    print("開始訓練模型...")
    
    # 設置混合精度訓練
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    
    # 設置分佈式訓練策略
    strategy = tf.distribute.MirroredStrategy()
    
    with strategy.scope():
        # 重新編譯模型以使用分佈式訓練
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss={
                'policy': 'categorical_crossentropy',
                'value': 'mse'
            },
            metrics={
                'policy': 'accuracy',
                'value': 'mae'
            }
        )
    
    # 創建數據集
    train_dataset = create_dataset(train_data, batch_size)
    val_dataset = create_dataset(val_data, batch_size)
    
    # 設置回調函數
    callbacks = [
        # 學習率調度器
        tf.keras.callbacks.LearningRateScheduler(
            lambda epoch, lr: lr * 0.9 if epoch > 10 else lr
        ),
        # 早停
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),
        # 模型檢查點
        tf.keras.callbacks.ModelCheckpoint(
            'best_cnn_rnn_model.h5',
            monitor='val_loss',
            save_best_only=True
        ),
        # TensorBoard日誌
        tf.keras.callbacks.TensorBoard(
            log_dir='./logs',
            update_freq='epoch'
        )
    ]
    
    # 訓練模型
    history = model.fit(
        train_dataset,
        epochs=epochs,
        validation_data=val_dataset,
        callbacks=callbacks,
        verbose=1
    )
    
    # 繪製訓練歷史
    plot_training_history(history)
    
    return model, history

def plot_training_history(history):
    plt.figure(figsize=(12, 4))
    
    # 繪製損失
    plt.subplot(1, 2, 1)
    plt.plot(history.history['policy_loss'], label='Policy Loss')
    plt.plot(history.history['value_loss'], label='Value Loss')
    plt.plot(history.history['val_policy_loss'], label='Val Policy Loss')
    plt.plot(history.history['val_value_loss'], label='Val Value Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # 繪製指標
    plt.subplot(1, 2, 2)
    plt.plot(history.history['policy_accuracy'], label='Policy Accuracy')
    plt.plot(history.history['value_mae'], label='Value MAE')
    plt.plot(history.history['val_policy_accuracy'], label='Val Policy Accuracy')
    plt.plot(history.history['val_value_mae'], label='Val Value MAE')
    plt.title('Model Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Metric')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()
def load_game_data(file_path, sequence_length=8):
    """
    載入圍棋對局數據並處理成適合CNN-RNN模型的格式
    
    Parameters:
    file_path: str - NPZ文件路徑
    sequence_length: int - RNN序列長度
    
    Returns:
    tuple - (X_board, X_sequence, y_policy, y_value)
    """
    print(f"正在載入遊戲數據從 {file_path}...")
    
    try:
        data = np.load(file_path)
        raw_X = data['X']  # 棋盤狀態
        raw_y = data['y']  # 移動位置
        winners = data.get('winners', None)  # 獲勝者信息
        
        num_games = len(raw_X)
        print(f"找到 {num_games} 個棋局記錄")
        
        # 初始化數組
        X_board = np.zeros((num_games, 19, 19, 3), dtype=np.float32)
        X_sequence = np.zeros((num_games, sequence_length, 19*19*3), dtype=np.float32)
        y_policy = np.zeros((num_games, 361), dtype=np.float32)
        y_value = np.zeros((num_games, 1), dtype=np.float32)
        
        # 重建遊戲歷史
        game_history = []
        current_game = []
        current_board = np.zeros((19, 19), dtype=np.int8)
        
        for i in range(num_games):
            if i > 0 and not np.array_equal(raw_X[i], current_board):
                # 新遊戲開始
                game_history.append(current_game)
                current_game = []
                current_board = np.zeros((19, 19), dtype=np.int8)
            
            # 更新當前遊戲歷史
            current_game.append(raw_X[i].copy())
            current_board = raw_X[i]
        
        # 添加最後一個遊戲
        if current_game:
            game_history.append(current_game)
        
        print(f"處理了 {len(game_history)} 個完整棋局")
        
        # 處理每個位置的特徵
        processed_positions = 0
        for game_idx, game in enumerate(game_history):
            for move_idx in range(len(game)):
                if processed_positions >= num_games:
                    break
                    
                # 處理棋盤特徵
                current_board = game[move_idx]
                current_player = 1 if np.sum(current_board == 1) == np.sum(current_board == -1) else -1
                
                X_board[processed_positions, :, :, 0] = (current_board == 1)  # 黑子
                X_board[processed_positions, :, :, 1] = (current_board == -1)  # 白子
                X_board[processed_positions, :, :, 2] = current_player  # 當前玩家
                
                # 處理序列特徵
                sequence_start = max(0, move_idx - sequence_length + 1)
                sequence_boards = game[sequence_start:move_idx + 1]
                
                # 填充序列至指定長度
                while len(sequence_boards) < sequence_length:
                    sequence_boards.insert(0, np.zeros_like(current_board))
                
                # 創建序列特徵
                for seq_idx, board in enumerate(sequence_boards[-sequence_length:]):
                    features = np.zeros(19*19*3, dtype=np.float32)
                    features[:19*19] = (board == 1).flatten()  # 黑子
                    features[19*19:2*19*19] = (board == -1).flatten()  # 白子
                    features[2*19*19:] = current_player  # 當前玩家
                    X_sequence[processed_positions, seq_idx] = features
                
                # 處理策略標籤(下一步移動)
                if move_idx + 1 < len(game):
                    next_board = game[move_idx + 1]
                    move_pos = np.where((next_board - current_board) != 0)
                    if len(move_pos[0]) > 0:
                        move_idx = move_pos[0][0] * 19 + move_pos[1][0]
                        y_policy[processed_positions, move_idx] = 1
                
                # 處理價值標籤(獲勝者)
                if winners is not None:
                    y_value[processed_positions] = winners[processed_positions] * current_player
                
                processed_positions += 1
                if processed_positions % 1000 == 0:
                    print(f"已處理 {processed_positions} 個位置")
        
        print("數據載入和處理完成")
        print(f"最終處理的位置數量: {processed_positions}")
        
        # 裁剪數組到實際大小
        X_board = X_board[:processed_positions]
        X_sequence = X_sequence[:processed_positions]
        y_policy = y_policy[:processed_positions]
        y_value = y_value[:processed_positions]
        
        # 數據驗證
        assert not np.any(np.isnan(X_board)), "X_board contains NaN values"
        assert not np.any(np.isnan(X_sequence)), "X_sequence contains NaN values"
        assert not np.any(np.isnan(y_policy)), "y_policy contains NaN values"
        assert not np.any(np.isnan(y_value)), "y_value contains NaN values"
        
        return X_board, X_sequence, y_policy, y_value
    
    except Exception as e:
        print(f"載入數據時發生錯誤: {str(e)}")
        raise

def verify_data(X_board, X_sequence, y_policy, y_value):
    """
    驗證載入的數據格式和內容是否正確
    """
    print("\n正在驗證數據...")
    
    # 檢查數據維度
    print(f"Board shape: {X_board.shape}")
    print(f"Sequence shape: {X_sequence.shape}")
    print(f"Policy shape: {y_policy.shape}")
    print(f"Value shape: {y_value.shape}")
    
    # 檢查數據範圍
    print(f"\nBoard value range: [{X_board.min()}, {X_board.max()}]")
    print(f"Sequence value range: [{X_sequence.min()}, {X_sequence.max()}]")
    print(f"Policy value range: [{y_policy.min()}, {y_policy.max()}]")
    print(f"Value value range: [{y_value.min()}, {y_value.max()}]")
    
    # 檢查政策標籤的有效性
    policy_sums = np.sum(y_policy, axis=1)
    print(f"\nPolicy sums statistics:")
    print(f"Mean: {policy_sums.mean():.4f}")
    print(f"Std: {policy_sums.std():.4f}")
    print(f"Min: {policy_sums.min():.4f}")
    print(f"Max: {policy_sums.max():.4f}")
    
    # 檢查序列數據的連續性
    sequence_diffs = np.diff(X_sequence, axis=1)
    print(f"\nSequence continuity:")
    print(f"Mean diff: {np.mean(np.abs(sequence_diffs)):.4f}")
    print(f"Max diff: {np.max(np.abs(sequence_diffs)):.4f}")
    
    print("\n數據驗證完成")
# 主訓練循環
def main():
    # 創建模型
    model = create_cnn_rnn_model()
    
    # 載入數據
    print("載入訓練數據...")
    # 假設我們有載入數據的函數
    X_board, X_seq, y_policy, y_value = load_game_data("path_to_your_data.npz")
    
    # 分割訓練集和驗證集
    split_idx = int(len(X_board) * 0.9)
    train_data = (
        X_board[:split_idx], 
        X_seq[:split_idx],
        y_policy[:split_idx], 
        y_value[:split_idx]
    )
    val_data = (
        X_board[split_idx:], 
        X_seq[split_idx:],
        y_policy[split_idx:], 
        y_value[split_idx:]
    )
    
    # 訓練迭代
    for iteration in tqdm(range(20), desc="Training Iterations"):
        # 訓練模型
        model, history = train_model(
            model,
            train_data,
            val_data,
            epochs=30,
            batch_size=2048
        )
        
        # 使用A*搜索生成新的訓練數據
        astar = AStarSearch(model)
        new_game_data = []
        
        print("生成自我對弈數據...")
        for _ in tqdm(range(500), desc="Self-play games"):
            game = GoGame()
            game_states = []
            
            while not game.is_game_over():
                move = astar.search(game)
                current_state = {
                    'board_state': game.board.copy(),
                    'move_history': game.history.copy(),
                    'next_move': move
                }
                game_states.append(current_state)
                game.play(move)
            
            # 記錄獲勝者
            winner = game.get_winner()[0]
            for state in game_states:
                state['winner'] = winner
                new_game_data.append(state)
        
        # 處理新數據
        X_new_board, X_new_seq, y_new_policy, y_new_value = preprocess_data(new_game_data)
        
        # 合併新舊數據
        X_board = np.concatenate([train_data[0], X_new_board])
        X_seq = np.concatenate([train_data[1], X_new_seq])
        y_policy = np.concatenate([train_data[2], y_new_policy])
        y_value = np.concatenate([train_data[3], y_new_value])
        
        # 更新訓練數據
        train_data = (X_board, X_seq, y_policy, y_value)
        
        # 保存模型
        model.save(f'cnn_rnn_model_iteration_{iteration+1}.h5')
        
        # 每5次迭代評估模型
        if (iteration + 1) % 5 == 0:
            evaluate_model(model, val_data)

def evaluate_model(model, val_data):
    """評估模型性能"""
    X_val_board, X_val_seq, y_val_policy, y_val_value = val_data
    results = model.evaluate(
        [X_val_board, X_val_seq],
        {'policy': y_val_policy, 'value': y_val_value},
        verbose=1
    )
    print("\nValidation Results:")
    for metric, value in zip(model.metrics_names, results):
        print(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    main()