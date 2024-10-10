import re
import numpy as np
import os
from tqdm import tqdm

def read_sgf_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def sgf_to_numpy(sgf_content, board_size=19):
    # 解析SGF內容
    moves = re.findall(r';([BW])\[([a-s]{2})\]', sgf_content)
    handicap = re.findall(r'AB(?:\[([a-s]{2})\])+', sgf_content)
    
    # 創建一個空的NumPy陣列列表來存儲每一步的棋盤狀態
    boards = [np.zeros((board_size, board_size), dtype=int)]
    
    # 處理讓子
    if handicap:
        for handicap_move in handicap[0].split(']['):
            x = ord(handicap_move[0]) - ord('a')
            y = ord(handicap_move[1]) - ord('a')
            boards[0][y, x] = 1
    
    # 填充陣列
    for color, move in moves:
        x = ord(move[0]) - ord('a')
        y = ord(move[1]) - ord('a')
        value = 1 if color == 'B' else -1  # 1 代表黑子, -1 代表白子
        new_board = boards[-1].copy()
        new_board[y, x] = value
        boards.append(new_board)
    
    return boards

def process_sgf_file(file_path, board_size=19):
    sgf_content = read_sgf_file(file_path)
    return sgf_to_numpy(sgf_content, board_size)

def process_kgs_data(main_directory, output_file):
    all_game_states = []
    all_next_moves = []
    error_files = []

    for root, dirs, files in os.walk(main_directory):
        for file in tqdm(files, desc=f"處理 {root}", leave=False):
            if file.endswith('.sgf'):
                file_path = os.path.join(root, file)
                try:
                    board_states = process_sgf_file(file_path)
                    
                    for i in range(len(board_states) - 1):
                        all_game_states.append(board_states[i])
                        next_move = np.zeros((19, 19), dtype=np.int8)
                        diff = board_states[i+1] - board_states[i]
                        next_move[diff != 0] = 1
                        all_next_moves.append(next_move)
                except Exception as e:
                    print(f"處理文件時出錯 {file_path}：{str(e)}")
                    error_files.append((file_path, str(e)))

    if all_game_states and all_next_moves:
        X = np.array(all_game_states)
        y = np.array(all_next_moves)
        
        np.savez_compressed(output_file, X=X, y=y)
        print(f"完成處理 {len(X)} 個棋局狀態。已保存到 {output_file}")
    else:
        print("警告：沒有處理到任何有效的棋局數據。")
    
    if error_files:
        print("\n處理過程中出錯的文件：")
        for file, error in error_files:
            print(f"{file}: {error}")

def main():
    main_directory = r"C:\Users\User\Desktop\gogamedata8" # data path
    output_file = "processed_kgs_data8.npz"
    process_kgs_data(main_directory, output_file)
    print("處理完成")

if __name__ == "__main__":
    main()