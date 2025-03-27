import os
import re
import numpy as np
from tqdm import tqdm
import random

def process_kgs_dataset(main_directory, output_file, min_rank=7, sample_rate=0.7, max_files=None):
    """處理KGS圍棋數據集中的SGF文件
    
    Args:
        main_directory: KGS數據主目錄
        output_file: 輸出的NPZ文件路徑
        min_rank: 最低段位過濾(默認7段以上)
        sample_rate: 高段位比例
        max_files: 最大處理文件數
    """
    # 收集所有SGF文件路徑
    all_files = []
    for root, dirs, files in os.walk(main_directory):
        for file in files:
            if file.endswith('.sgf'):
                all_files.append(os.path.join(root, file))
    
    # 顯示找到的文件數量
    print(f"找到 {len(all_files)} 個SGF文件")
    
    # 隨機打亂文件順序
    random.shuffle(all_files)
    
    # 限制文件數量
    if max_files is not None:
        all_files = all_files[:max_files]
        print(f"限制處理前 {max_files} 個文件")
    
    # 處理數據
    all_game_states = []
    all_next_moves = []
    all_game_results = []  # 存儲勝負信息
    high_rank_count = 0
    low_rank_count = 0
    error_files = []
    
    for file_path in tqdm(all_files, desc="處理SGF文件"):
        try:
            # 讀取SGF
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                sgf_content = f.read()
            
            # 提取段位信息
            rank = get_player_rank(sgf_content)
            
            # 提取勝負信息
            game_result = get_game_result(sgf_content)
            
            # 根據段位和目前的高低段位比例決定是否處理此文件
            current_ratio = high_rank_count / (high_rank_count + low_rank_count + 1e-10)
            if (rank >= min_rank and current_ratio < sample_rate) or \
               (rank < min_rank and current_ratio >= sample_rate):
                
                # 處理棋盤狀態
                board_states = sgf_to_numpy(sgf_content)
                
                # 添加到數據集
                for i in range(len(board_states) - 1):
                    all_game_states.append(board_states[i])
                    next_move = np.zeros((19, 19), dtype=np.int8)
                    diff = board_states[i+1] - board_states[i]
                    next_move[diff != 0] = 1
                    all_next_moves.append(next_move)
                    
                    # 計算當前玩家
                    black_count = np.sum(board_states[i] == 1)
                    white_count = np.sum(board_states[i] == -1)
                    current_player = 1 if black_count <= white_count else -1
                    
                    # 從當前玩家角度添加勝負結果
                    all_game_results.append(game_result * current_player)
                
                # 更新計數
                if rank >= min_rank:
                    high_rank_count += 1
                else:
                    low_rank_count += 1
        
        except Exception as e:
            error_files.append((file_path, str(e)))
            continue
    
    # 保存處理後的數據
    if all_game_states and all_next_moves:
        X = np.array(all_game_states)
        y = np.array(all_next_moves)
        winners = np.array(all_game_results)
        
        np.savez_compressed(output_file, X=X, y=y, winners=winners)
        print(f"完成處理 {len(X)} 個棋局狀態。已保存到 {output_file}")
        print(f"高段位對局數: {high_rank_count}, 低段位對局數: {low_rank_count}")
        print(f"實際比例: {high_rank_count / (high_rank_count + low_rank_count):.2f}")
    else:
        print("警告：沒有處理到任何有效的棋局數據。")
    
    # 報告錯誤
    if error_files:
        print(f"\n處理過程中有 {len(error_files)} 個文件出錯")
        with open("error_files.log", "w") as f:
            for file, error in error_files:
                f.write(f"{file}: {error}\n")

def get_player_rank(sgf_content):
    """從SGF內容中提取棋手段位"""
    # 提取黑棋段位
    br_match = re.search(r'BR\[([^\]]+)\]', sgf_content)
    if br_match:
        rank_str = br_match.group(1)
        
        # 處理段位
        if 'p' in rank_str.lower():  # 職業
            num_match = re.search(r'\d+', rank_str)
            return int(num_match.group(0)) + 10 if num_match else 10
        elif 'd' in rank_str.lower():  # 業餘段
            num_match = re.search(r'\d+', rank_str)
            return int(num_match.group(0)) if num_match else 1
        elif 'k' in rank_str.lower():  # 業餘級
            return 0
    
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
    
    # 找不到任何段位信息
    return 0

def get_game_result(sgf_content):
    """從SGF中提取比賽結果：1表示黑勝，-1表示白勝，0表示平局"""
    result_match = re.search(r'RE\[([^\]]+)\]', sgf_content)
    if result_match:
        result_str = result_match.group(1)
        if 'B+' in result_str:
            return 1  # 黑勝
        elif 'W+' in result_str:
            return -1  # 白勝
        else:
            return 0  # 平局或無效記錄
    return 0  # 找不到勝負信息

def sgf_to_numpy(sgf_content, board_size=19):
    """將SGF轉換為一系列棋盤狀態NumPy數組"""
    # 提取讓子
    handicap_stones = []
    handicap_match = re.search(r'AB(?:\[([a-s][a-s])\])+', sgf_content)
    if handicap_match:
        handicap_str = handicap_match.group(0)
        handicap_stones = re.findall(r'\[([a-s][a-s])\]', handicap_str)
    
    # 提取所有移動
    moves = re.findall(r';([BW])\[([a-s][a-s])\]', sgf_content)
    
    # 初始棋盤
    board = np.zeros((board_size, board_size), dtype=np.int8)
    
    # 應用讓子
    for stone in handicap_stones:
        x, y = ord(stone[0]) - ord('a'), ord(stone[1]) - ord('a')
        board[y, x] = 1  # 黑子
    
    # 記錄棋盤狀態序列
    board_states = [board.copy()]
    
    # 應用移動
    for color, coord in moves:
        if coord:  # 跳過虛手
            x, y = ord(coord[0]) - ord('a'), ord(coord[1]) - ord('a')
            if 0 <= x < board_size and 0 <= y < board_size:
                new_board = board.copy()
                new_board[y, x] = 1 if color == 'B' else -1
                board = new_board
                board_states.append(board.copy())
    
    return board_states

# 使用示例
if __name__ == "__main__":
    main_directory = r"C:\Users\User\Desktop\Gogame data"
    output_file = "process_kgs_dataset_10.npz"
    process_kgs_dataset(main_directory, output_file, min_rank=7, sample_rate=0.1, max_files=10)
    print("處理完成")