import numpy as np
import os
import re
import glob
import random
from tqdm import tqdm
from game.go_game import GoGame
import tensorflow as tf

def read_sgf_file(file_path, encoding='utf-8'):
    """讀取SGF文件內容。"""
    try:
        with open(file_path, 'r', encoding=encoding) as file:
            return file.read()
    except UnicodeDecodeError:
        # 如果UTF-8失敗，嘗試其他編碼
        try:
            with open(file_path, 'r', encoding='latin-1') as file:
                return file.read()
        except Exception as e:
            print(f"無法讀取文件 {file_path}: {str(e)}")
            return None

def get_player_rank(sgf_content):
    """從SGF內容中提取棋手段位。"""
    # 提取黑棋段位
    br_match = re.search(r'BR\[([^\]]+)\]', sgf_content)
    if br_match:
        rank_str = br_match.group(1)
        
        # 處理段位
        if 'p' in rank_str.lower():  # 職業
            # 提取數字部分
            num_match = re.search(r'\d+', rank_str)
            return int(num_match.group(0)) + 10 if num_match else 10
        elif 'd' in rank_str.lower():  # 業餘段
            # 提取數字部分
            num_match = re.search(r'\d+', rank_str)
            return int(num_match.group(0)) if num_match else 1
        elif 'k' in rank_str.lower():  # 業餘級
            # 提取數字部分
            num_match = re.search(r'\d+', rank_str)
            return -int(num_match.group(0)) if num_match else -30
    
    # 如果找不到段位，嘗試提取白棋段位
    wr_match = re.search(r'WR\[([^\]]+)\]', sgf_content)
    if wr_match:
        rank_str = wr_match.group(1)
        
        # 處理段位
        if 'p' in rank_str.lower():  # 職業
            num_match = re.search(r'\d+', rank_str)
            return int(num_match.group(0)) + 10 if num_match else 10
        elif 'd' in rank_str.lower():  # 業餘段
            num_match = re.search(r'\d+', rank_str)
            return int(num_match.group(0)) if num_match else 1
        elif 'k' in rank_str.lower():  # 業餘級
            num_match = re.search(r'\d+', rank_str)
            return -int(num_match.group(0)) if num_match else -30
    
    # 找不到任何段位信息
    return 0

def sgf_to_game_states(sgf_content, min_moves=0, max_moves=None, board_size=19):
    """將SGF轉換為一系列遊戲狀態。
    
    Args:
        sgf_content: SGF文件內容
        min_moves: 最小移動數（用於跳過開局）
        max_moves: 最大移動數（用於限制遊戲長度）
        board_size: 棋盤大小
        
    Returns:
        tuple: (game_states, next_moves, winner)
    """
    if not sgf_content:
        return [], [], None
    
    # 提取勝負信息
    result_match = re.search(r'RE\[([^\]]+)\]', sgf_content)
    winner = None
    if result_match:
        result_str = result_match.group(1)
        if 'B+' in result_str:
            winner = 1  # 黑勝
        elif 'W+' in result_str:
            winner = -1  # 白勝
        elif 'Draw' in result_str or '0' in result_str:
            winner = 0  # 平局
    
    # 提取讓子
    handicap = 0
    handicap_match = re.search(r'HA\[(\d+)\]', sgf_content)
    if handicap_match:
        handicap = int(handicap_match.group(1))
    
    # 提取棋盤大小
    size_match = re.search(r'SZ\[(\d+)\]', sgf_content)
    if size_match:
        board_size = int(size_match.group(1))
    
    # 提取移動
    moves = re.findall(r';([BW])\[([a-zA-Z]{0,2})\]', sgf_content)
    
    # 處理讓子情況
    game = GoGame(size=board_size)
    if handicap > 1:
        # 提取讓子點位
        handicap_points = re.findall(r'AB(?:\[([a-zA-Z]{2})\])+', sgf_content)
        if handicap_points:
            for point in handicap_points:
                x = ord(point[0].lower()) - ord('a')
                y = ord(point[1].lower()) - ord('a')
                if 0 <= x < board_size and 0 <= y < board_size:
                    game.board[x, y] = 1
            
            # 讓子時，白方先行
            game.current_player = -1
            game.history.append(game.board.copy())
    
    # 存儲遊戲狀態
    game_states = []
    next_moves = []
    
    # 跟踪當前移動數
    move_count = 0
    
    # 應用所有移動
    for color, coord in moves:
        # 儲存當前狀態和下一步移動
        if move_count >= min_moves and (max_moves is None or move_count < max_moves):
            # 跳過存儲虛手
            if coord and coord.lower() != 'tt':
                current_state = game.get_state()
                x = ord(coord[0].lower()) - ord('a')
                y = ord(coord[1].lower()) - ord('a')
                
                # 確保座標在棋盤範圍內
                if 0 <= x < board_size and 0 <= y < board_size:
                    game_states.append(current_state)
                    next_moves.append((x, y))
        
        # 應用移動到遊戲中
        if not coord or coord.lower() == 'tt':
            # 處理虛手
            move = 'pass'
        else:
            x = ord(coord[0].lower()) - ord('a')
            y = ord(coord[1].lower()) - ord('a')
            move = (x, y)
        
        # 確保移動有效
        if move == 'pass' or (0 <= move[0] < board_size and 0 <= move[1] < board_size):
            game.play(move)
            move_count += 1
    
    return game_states, next_moves, winner

def convert_game_states_to_features(game_states, next_moves, winner, num_history=8):
    """將遊戲狀態轉換為神經網絡輸入特徵。
    
    Args:
        game_states: 遊戲狀態列表（來自GoGame.get_state()）
        next_moves: 對應的下一步移動列表
        winner: 遊戲勝者
        num_history: 歷史平面數
        
    Returns:
        tuple: (X_board, y_policy, y_value)
    """
    if not game_states:
        return [], [], []
    
    # 確定棋盤大小
    game = GoGame()
    game.set_state(game_states[0])
    board_size = game.size
    
    # 預分配數組
    X_board = []
    y_policy = []
    y_value = []
    
    for i, (state, next_move) in enumerate(zip(game_states, next_moves)):
        # 設置遊戲狀態
        game = GoGame()
        game.set_state(state)
        
        # 獲取特徵平面
        features = game.get_state_features(num_history)
        X_board.append(features)
        
        # 建立策略標籤（one-hot）
        policy = np.zeros(board_size * board_size, dtype=np.float32)
        move_idx = next_move[0] * board_size + next_move[1]
        policy[move_idx] = 1.0
        y_policy.append(policy)
        
        # 建立價值標籤
        if winner is not None:
            # 轉換為當前玩家的觀點
            value = winner * game.current_player
            y_value.append(value)
        else:
            # 如果沒有勝者信息，使用0（表示未知）
            y_value.append(0)
    
    return np.array(X_board), np.array(y_policy), np.array(y_value, dtype=np.float32)

def process_sgf_files(file_patterns, output_file=None, min_rank=7, max_files=None, 
                     min_moves=30, hist_planes=8, train_ratio=0.9, verbose=True):
    """批量處理SGF文件並生成訓練數據。
    
    Args:
        file_patterns: SGF文件路徑模式（可以是列表或單個字符串）
        output_file: 輸出NPZ文件路徑
        min_rank: 最低棋手段位，低於此段位的棋譜將被忽略
        max_files: 最大處理文件數
        min_moves: 最小移動數
        hist_planes: 歷史特徵平面數
        train_ratio: 訓練集比例
        verbose: 是否顯示進度條
        
    Returns:
        dict: 訓練和驗證數據
    """
    all_files = []
    
    # 收集所有符合模式的文件
    if isinstance(file_patterns, list):
        for pattern in file_patterns:
            all_files.extend(glob.glob(pattern))
    else:
        all_files = glob.glob(file_patterns)
    
    # 隨機打亂文件順序
    random.shuffle(all_files)
    
    # 限制文件數量
    if max_files is not None and max_files > 0:
        all_files = all_files[:max_files]
    
    # 用於收集數據
    X_boards = []
    y_policies = []
    y_values = []
    
    # 計數器
    total_files = 0
    skipped_files = 0
    total_positions = 0
    
    # 處理文件
    iterator = tqdm(all_files) if verbose else all_files
    for file_path in iterator:
        # 讀取SGF文件
        sgf_content = read_sgf_file(file_path)
        if not sgf_content:
            skipped_files += 1
            continue
        
        # 檢查棋手段位
        rank = get_player_rank(sgf_content)
        if rank < min_rank:
            skipped_files += 1
            continue
        
        # 轉換為遊戲狀態
        try:
            game_states, next_moves, winner = sgf_to_game_states(
                sgf_content, min_moves=min_moves)
            
            # 確保有足夠的遊戲狀態
            if len(game_states) < min_moves:
                skipped_files += 1
                continue
            
            # 轉換為神經網絡輸入
            X_board, y_policy, y_value = convert_game_states_to_features(
                game_states, next_moves, winner, num_history=hist_planes)
            
            # 添加到數據集
            X_boards.append(X_board)
            y_policies.append(y_policy)
            y_values.append(y_value)
            
            # 更新計數器
            total_files += 1
            total_positions += len(game_states)
            
        except Exception as e:
            if verbose:
                print(f"處理文件 {file_path} 時出錯: {str(e)}")
            skipped_files += 1
    
    # 將所有數據合併為單個數組
    if X_boards:
        X_board = np.concatenate(X_boards, axis=0)
        y_policy = np.concatenate(y_policies, axis=0)
        y_value = np.concatenate(y_values, axis=0)
    else:
        # 如果沒有數據，創建空數組
        X_board = np.array([])
        y_policy = np.array([])
        y_value = np.array([])
    
    # 顯示統計信息
    if verbose:
        print(f"處理了 {total_files} 個文件，跳過了 {skipped_files} 個文件")
        print(f"總共生成了 {total_positions} 個棋局位置")
        if X_board.size > 0:
            print(f"數據形狀：X_board: {X_board.shape}, y_policy: {y_policy.shape}, y_value: {y_value.shape}")
    
    # 分割訓練集和驗證集
    if X_board.size > 0:
        split_idx = int(len(X_board) * train_ratio)
        
        train_data = {
            'X_board': X_board[:split_idx],
            'y_policy': y_policy[:split_idx],
            'y_value': y_value[:split_idx]
        }
        
        val_data = {
            'X_board': X_board[split_idx:],
            'y_policy': y_policy[split_idx:],
            'y_value': y_value[split_idx:]
        }
        
        # 保存數據到文件
        if output_file:
            np.savez_compressed(
                output_file,
                X_train=train_data['X_board'],
                y_policy_train=train_data['y_policy'],
                y_value_train=train_data['y_value'],
                X_val=val_data['X_board'],
                y_policy_val=val_data['y_policy'],
                y_value_val=val_data['y_value']
            )
            
            if verbose:
                print(f"數據已保存到 {output_file}")
        
        return train_data, val_data
    
    # 如果沒有數據，返回空字典
    if verbose:
        print("警告：沒有生成任何有效數據")
    
    empty_data = {'X_board': np.array([]), 'y_policy': np.array([]), 'y_value': np.array([])}
    return empty_data, empty_data

def load_data(file_path, verbose=True):
    """從NPZ文件載入訓練數據。"""
    if verbose:
        print(f"從 {file_path} 載入數據...")
    
    try:
        data = np.load(file_path)
        
        # 提取訓練集和驗證集
        train_data = {
            'X_board': data['X_train'],
            'y_policy': data['y_policy_train'],
            'y_value': data['y_value_train']
        }
        
        val_data = {
            'X_board': data['X_val'],
            'y_policy': data['y_policy_val'],
            'y_value': data['y_value_val']
        }
        
        if verbose:
            print(f"訓練數據形狀：X_board: {train_data['X_board'].shape}, " +
                 f"y_policy: {train_data['y_policy'].shape}, " +
                 f"y_value: {train_data['y_value'].shape}")
            print(f"驗證數據形狀：X_board: {val_data['X_board'].shape}, " +
                 f"y_policy: {val_data['y_policy'].shape}, " +
                 f"y_value: {val_data['y_value'].shape}")
        
        return train_data, val_data
    
    except Exception as e:
        if verbose:
            print(f"載入數據時出錯: {str(e)}")
        return None, None

def create_tf_dataset(data, batch_size=32, shuffle_buffer=10000, repeat=True, augment=True):
    """創建TensorFlow數據集。
    
    Args:
        data: 包含'X_board', 'y_policy', 'y_value'的字典
        batch_size: 批次大小
        shuffle_buffer: 洗牌緩衝區大小
        repeat: 是否重複數據集
        augment: 是否進行數據增強
        
    Returns:
        tf.data.Dataset: TensorFlow數據集
    """
    X_board = data['X_board']
    y_policy = data['y_policy']
    y_value = data['y_value']
    
    # 確保數據不為空
    if X_board.size == 0:
        return None
    
    # 創建數據集
    dataset = tf.data.Dataset.from_tensor_slices(
        (X_board, {'policy': y_policy, 'value': y_value})
    )
    
    # 數據增強
    if augment:
        dataset = dataset.map(augment_data, num_parallel_calls=tf.data.AUTOTUNE)
    
    # 洗牌和批次處理
    if shuffle_buffer > 0:
        dataset = dataset.shuffle(buffer_size=min(shuffle_buffer, len(X_board)))
    
    dataset = dataset.batch(batch_size)
    
    # 是否重複
    if repeat:
        dataset = dataset.repeat()
    
    # 預取以提高性能
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

def augment_data(x, y):
    """對圍棋數據進行增強（旋轉和翻轉）。"""
    # 隨機選擇變換：0-7
    k = tf.random.uniform([], minval=0, maxval=8, dtype=tf.int32)
    
    # 提取特徵和標籤
    features = x
    policy = y['policy']
    value = y['value']
    
    # 獲取棋盤大小
    board_size = tf.shape(features)[1]  # 通常是19
    
    # 應用變換到特徵
    transformed_features = tf.cond(
        k < 4,
        lambda: tf.image.rot90(features, k=k),
        lambda: tf.image.flip_left_right(tf.image.rot90(features, k=k-4))
    )
    
    # 應用變換到策略
    policy_2d = tf.reshape(policy, [board_size, board_size])
    transformed_policy = tf.cond(
        k < 4,
        lambda: tf.image.rot90(policy_2d, k=k),
        lambda: tf.image.flip_left_right(tf.image.rot90(policy_2d, k=k-4))
    )
    transformed_policy = tf.reshape(transformed_policy, [-1])
    
    return transformed_features, {'policy': transformed_policy, 'value': value}