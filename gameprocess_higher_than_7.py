import re
import numpy as np
import os
from tqdm import tqdm
import random

def read_sgf_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def get_player_rank(sgf_content):
    rank_match = re.search(r'BR\[([^\]]+)\]', sgf_content)
    if rank_match:
        rank = rank_match.group(1)
        if 'k' in rank.lower():
            return 0
        elif 'd' in rank.lower():
            dan = int(rank[:-1])
            return dan if dan >= 7 else 0
    return 0

def sgf_to_numpy(sgf_content, board_size=19):
    moves = re.findall(r';([BW])\[([a-s]{2})\]', sgf_content)
    handicap = re.findall(r'AB(?:\[([a-s]{2})\])+', sgf_content)
    
    boards = [np.zeros((board_size, board_size), dtype=int)]
    
    if handicap:
        for handicap_move in handicap[0].split(']['):
            x, y = ord(handicap_move[0]) - ord('a'), ord(handicap_move[1]) - ord('a')
            boards[0][y, x] = 1
    
    for color, move in moves:
        x, y = ord(move[0]) - ord('a'), ord(move[1]) - ord('a')
        value = 1 if color == 'B' else -1
        new_board = boards[-1].copy()
        new_board[y, x] = value
        boards.append(new_board)
    
    return boards

def process_sgf_file(file_path, board_size=19):
    sgf_content = read_sgf_file(file_path)
    rank = get_player_rank(sgf_content)
    boards = sgf_to_numpy(sgf_content, board_size)
    return boards, rank

def process_kgs_data(main_directory, output_file, high_rank_ratio=0.7):
    all_game_states = []
    all_next_moves = []
    high_rank_count = 0
    low_rank_count = 0
    error_files = []

    all_files = []
    for root, dirs, files in os.walk(main_directory):
        for file in files:
            if file.endswith('.sgf'):
                all_files.append(os.path.join(root, file))

    random.shuffle(all_files)

    for file_path in tqdm(all_files, desc="處理文件"):
        try:
            board_states, rank = process_sgf_file(file_path)
            
            current_ratio = high_rank_count / (high_rank_count + low_rank_count + 1e-10)
            if (rank >= 7 and current_ratio < high_rank_ratio) or \
               (rank < 7 and current_ratio >= high_rank_ratio):
                
                for i in range(len(board_states) - 1):
                    all_game_states.append(board_states[i])
                    next_move = np.zeros((19, 19), dtype=np.int8)
                    diff = board_states[i+1] - board_states[i]
                    next_move[diff != 0] = 1
                    all_next_moves.append(next_move)
                
                if rank >= 7:
                    high_rank_count += 1
                else:
                    low_rank_count += 1
        
        except Exception as e:
            print(f"處理文件時出錯 {file_path}：{str(e)}")
            error_files.append((file_path, str(e)))

    if all_game_states and all_next_moves:
        X = np.array(all_game_states)
        y = np.array(all_next_moves)
        
        np.savez_compressed(output_file, X=X, y=y)
        print(f"完成處理 {len(X)} 個棋局狀態。已保存到 {output_file}")
        print(f"高段位對局數: {high_rank_count}, 低段位對局數: {low_rank_count}")
        print(f"實際比例: {high_rank_count / (high_rank_count + low_rank_count):.2f}")
    else:
        print("警告：沒有處理到任何有效的棋局數據。")
    
    if error_files:
        print("\n處理過程中出錯的文件：")
        for file, error in error_files:
            print(f"{file}: {error}")

def main():
    main_directory = r"C:\Users\User\Desktop\gogamedata8" # data path
    output_file = "processed_kgs_data8.npz"
    process_kgs_data(main_directory, output_file, high_rank_ratio=0.7)
    print("處理完成")

if __name__ == "__main__":
    main()