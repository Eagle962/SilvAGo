import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

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

# CNN 模型定義
def create_cnn_model(input_shape=(19, 19, 3)):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.Dropout(0.2)(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    policy_output = layers.Dense(361, activation='softmax', name='policy')(x)
    value_output = layers.Dense(1, activation='tanh', name='value')(x)
    model = models.Model(inputs=inputs, outputs=[policy_output, value_output])
    model.compile(optimizer='adam',
                loss={'policy': 'categorical_crossentropy', 'value': 'mse'},
                loss_weights={'policy': 1.0, 'value': 1.0},
                metrics={'policy': 'accuracy', 'value': 'mae'})
    return model

def create_dataset(data, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices(data)
    return dataset.shuffle(10000).batch(batch_size).prefetch(tf.data.AUTOTUNE)

def lr_schedule(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)


# MCTS 節點類
class MCTSNode:
    def __init__(self, game_state, parent=None, move=None):
        self.game_state = game_state
        self.parent = parent
        self.move = move
        self.children = {}
        self.visits = 0
        self.value = 0

# MCTS 實現
class MCTS:
    def __init__(self, model, num_simulations=1000, c_puct=1.0):
        self.model = model
        self.num_simulations = num_simulations
        self.c_puct = c_puct

    def search(self, root_state):
        root = MCTSNode(root_state)
        
        for _ in range(self.num_simulations):
            node = root
            game = GoGame()
            game.board = root_state.board.copy()
            game.current_player = root_state.current_player
            
            # Selection
            while node.children and not game.is_game_over():
                if not all(child.visits for child in node.children.values()):
                    node = self.expand(node, game)
                else:
                    node = self.select_child(node)
                    game.play(node.move)
            
            # Expansion
            if not game.is_game_over():
                node = self.expand(node, game)
            
            # Simulation
            value = self.simulate(game)
            
            # Backpropagation
            while node:
                node.visits += 1
                node.value += value if node.game_state.current_player == game.current_player else -value
                node = node.parent
        
        return max(root.children.items(), key=lambda x: x[1].visits)[0]

    def expand(self, node, game):
        legal_moves = game.get_legal_moves()
        for move in legal_moves:
            if move not in node.children:
                new_game = GoGame()
                new_game.board = game.board.copy()
                new_game.current_player = game.current_player
                new_game.play(move)
                node.children[move] = MCTSNode(new_game, parent=node, move=move)
        return random.choice(list(node.children.values()))

    def select_child(self, node):
        total_visits = sum(child.visits for child in node.children.values())
        return max(node.children.values(), key=lambda child: child.value / (child.visits + 1e-8) + self.c_puct * np.sqrt(total_visits) / (1 + child.visits))

    def simulate(self, game):
        while not game.is_game_over():
            move_probs = self.get_move_probabilities(game)
            move = np.random.choice(361, p=move_probs)
            game.play((move // 19, move % 19))
        return game.get_winner()[0]

    def get_move_probabilities(self, game):
        board_state = self.preprocess_board(game.board, game.current_player)
        move_probs = self.model.predict(np.expand_dims(board_state, axis=0))[0]
        legal_moves = game.get_legal_moves()
        legal_moves = [move for move in legal_moves if move != 'pass']
        mask = np.zeros(361)
        for move in legal_moves:
            mask[move[0] * 19 + move[1]] = 1
        masked_probs = move_probs * mask
        return masked_probs / np.sum(masked_probs)

    def preprocess_board(self, board, current_player):
        state = np.zeros((19, 19, 3), dtype=np.float32)
        state[:,:,0] = (board == current_player)
        state[:,:,1] = (board == -current_player)
        state[:,:,2] = current_player  # 當前玩家的顏色
        return state

# 自我對弈函數
def self_play(model, num_games=100):
    mcts = MCTS(model)
    training_data = []

    for _ in range(num_games):
        game = GoGame()
        game_states = []
        move_probs = []
        current_player = []

        while not game.is_game_over():
            board_state = mcts.preprocess_board(game.board, game.current_player)
            game_states.append(board_state)
            current_player.append(game.current_player)

            if random.random() < 0.05:  # 5% 的機會隨機走子，增加探索
                legal_moves = game.get_legal_moves()
                move = random.choice(legal_moves)
            else:
                move = mcts.search(game)

            probs = np.zeros(361)
            if move != 'pass':
                probs[move[0] * 19 + move[1]] = 1
            move_probs.append(probs)

            game.play(move)

        winner = game.get_winner()[0]
        for state, probs, player in zip(game_states, move_probs, current_player):
            training_data.append((state, probs, winner * player))

    return training_data

def load_game_data(file_path):
    print(f"正在載入遊戲數據從 {file_path}...")
    data = np.load(file_path)
    X = data['X']
    y = data['y']
    
    num_games = X.shape[0]
    X_processed = np.zeros((num_games, 19, 19, 3), dtype=np.float32)
    y_policy = np.zeros((num_games, 361), dtype=np.float32)
    y_value = np.zeros((num_games, 1), dtype=np.float32)
    
    for i in range(num_games):
        X_processed[i, :, :, 0] = (X[i] == 1)  # 黑子
        X_processed[i, :, :, 1] = (X[i] == -1)  # 白子
        X_processed[i, :, :, 2] = 1 if np.sum(X[i] == 1) == np.sum(X[i] == -1) else -1  # 當前玩家 (1 for 黑, -1 for 白)
        
        move = np.where(y[i] == 1)
        if move[0].size > 0 and move[1].size > 0:
            y_policy[i, move[0][0] * 19 + move[1][0]] = 1
        
        # 假設黑子獲勝為 1，白子獲勝為 -1
        y_value[i] = X_processed[i, :, :, 2].sum()
    
    print(f"載入完成。總共 {num_games} 個棋局狀態。")
    return X_processed, y_policy, y_value

def train_model_on_game_data(model, X, y_policy, y_value, epochs=10, batch_size=32):
    print("開始訓練模型...")
    history = model.fit(X, {'policy': y_policy, 'value': y_value}, 
                        epochs=epochs, batch_size=batch_size, validation_split=0.1,
                        callbacks=[tf.keras.callbacks.ProgbarLogger()])
    print("訓練完成。")
    return history

def preprocess_game_data(game_data):
    X = []
    y_move = []
    y_value = []
    for board_state, move, winner in game_data:
        X.append(board_state)
        move_prob = np.zeros(361)
        move_prob[move[0] * 19 + move[1]] = 1
        y_move.append(move_prob)
        y_value.append(winner)
    return np.array(X), np.array(y_move), np.array(y_value)


# 訓練循環
def train_model(model, train_data, val_data, epochs=30, batch_size=2048):
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                    loss={'policy': 'categorical_crossentropy', 'value': 'mse'},
                    metrics={'policy': 'accuracy', 'value': 'mae'})

    tf.keras.mixed_precision.set_global_policy('mixed_float16')

    train_dataset = create_dataset(train_data, batch_size)
    val_dataset = create_dataset(val_data, batch_size)

    callbacks = [
        tf.keras.callbacks.LearningRateScheduler(lr_schedule),
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint('best_model.h5', save_best_only=True),
        tf.keras.callbacks.TensorBoard(log_dir='./logs')
    ]

    history = model.fit(
        train_dataset,
        epochs=epochs,
        validation_data=val_dataset,
        callbacks=callbacks
    )

    plot_training_history(history)
    return model, history

def train_model_on_self_play(model, training_data, epochs=5, batch_size=32):
    states, policies, values = zip(*training_data)
    states = np.array(states)
    policies = np.array(policies)
    values = np.array(values)
    model.fit(states, [policies, values], epochs=epochs, batch_size=batch_size, validation_split=0.1)
    
def plot_training_history(history):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['policy_loss'], label='Policy Loss')
    plt.plot(history.history['value_loss'], label='Value Loss')
    plt.plot(history.history['val_policy_loss'], label='Val Policy Loss')
    plt.plot(history.history['val_value_loss'], label='Val Value Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
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
    
    
def main():
    # 載入和預處理數據
    X, y_policy, y_value = load_game_data("path_to_your_game_data.npz")
    
    # 分割數據為訓練集和驗證集
    split_index = int(0.9 * len(X))  # 90% 用於訓練，10% 用於驗證
    train_data = (X[:split_index], y_policy[:split_index], y_value[:split_index])
    val_data = (X[split_index:], y_policy[split_index:], y_value[split_index:])

    # 創建模型
    model = create_cnn_model()

    # 訓練模型
    model, history = train_model(model, train_data, val_data, epochs=30, batch_size=2048)

    # 繪製訓練歷史
    plot_training_history(history)

    # 保存最終模型
    model.save('final_go_ai_model.h5')

    # 自我對弈和微調（如果需要）
    for iteration in tqdm(range(20), desc="自我對弈迭代"):
        training_data = self_play(model, num_games=500)
        X_self, y_policy_self, y_value_self = preprocess_game_data(training_data)
        
        # 將自我對弈數據與原始訓練數據合併
        X_combined = np.concatenate([train_data[0], X_self])
        y_policy_combined = np.concatenate([train_data[1], y_policy_self])
        y_value_combined = np.concatenate([train_data[2], y_value_self])
        
        combined_data = (X_combined, y_policy_combined, y_value_combined)
        
        # 重新訓練模型
        model, history = train_model(model, combined_data, val_data, epochs=5, batch_size=2048)
        
        # 繪製每次迭代後的訓練歷史
        plot_training_history(history)

    # 保存最終模型
    model.save('SilvaGo1.h5')

if __name__ == "__main__":
    main()