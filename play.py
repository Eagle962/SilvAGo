#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
圍棋AI對弈腳本 - 允許使用者與AI對弈或觀看AI自我對弈
"""

import os
import sys
import argparse
import logging
import time
import numpy as np
import tensorflow as tf
from tqdm import tqdm

# 確保可以導入自定義模組
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 導入自定義模組
from game.go_game import GoGame
from ml.model import create_go_model, create_light_model, create_cnn_rnn_model
from search.mcts import MCTS

# 嘗試導入圖形界面庫
try:
    import pygame
    HAS_PYGAME = True
except ImportError:
    HAS_PYGAME = False
    print("未找到pygame庫，將使用命令行界面。使用 'pip install pygame' 安裝pygame以獲得圖形界面。")

# 設置日誌記錄
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('play.log')
    ]
)
logger = logging.getLogger(__name__)

# 定義顏色常量
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BOARD_COLOR = (220, 179, 92)  # 棋盤顏色
GRID_COLOR = (0, 0, 0)  # 格線顏色
HIGHLIGHT_COLOR = (255, 0, 0)  # 上一手標記顏色

class GoGUI:
    """圍棋圖形界面類。"""
    
    def __init__(self, game, cell_size=40, margin=40):
        """初始化圍棋界面。
        
        Args:
            game: GoGame實例
            cell_size: 單元格大小（像素）
            margin: 邊距（像素）
        """
        self.game = game
        self.cell_size = cell_size
        self.margin = margin
        self.board_size = game.size
        
        # 窗口大小
        self.window_size = 2 * margin + (self.board_size - 1) * cell_size
        
        # 初始化pygame
        pygame.init()
        self.font = pygame.font.SysFont('Arial', 16)
        self.screen = pygame.display.set_mode((self.window_size, self.window_size))
        pygame.display.set_caption('圍棋AI')
        
        # 加載棋子圖片（如果有）
        self.black_stone = None
        self.white_stone = None
        try:
            self.black_stone = pygame.image.load('assets/black_stone.png')
            self.black_stone = pygame.transform.scale(self.black_stone, (cell_size, cell_size))
            self.white_stone = pygame.image.load('assets/white_stone.png')
            self.white_stone = pygame.transform.scale(self.white_stone, (cell_size, cell_size))
        except:
            pass  # 使用圓圈作為棋子
        
        # 初始界面
        self.update_display()
    
    def update_display(self):
        """更新界面顯示。"""
        # 填充背景
        self.screen.fill(BOARD_COLOR)
        
        # 繪製棋盤格線
        for i in range(self.board_size):
            # 繪製水平線
            pygame.draw.line(
                self.screen, GRID_COLOR,
                (self.margin, self.margin + i * self.cell_size),
                (self.margin + (self.board_size - 1) * self.cell_size, self.margin + i * self.cell_size),
                2 if i == 0 or i == self.board_size - 1 else 1
            )
            # 繪製垂直線
            pygame.draw.line(
                self.screen, GRID_COLOR,
                (self.margin + i * self.cell_size, self.margin),
                (self.margin + i * self.cell_size, self.margin + (self.board_size - 1) * self.cell_size),
                2 if i == 0 or i == self.board_size - 1 else 1
            )
        
        # 繪製星位點
        star_points = []
        if self.board_size == 19:
            star_points = [(3, 3), (3, 9), (3, 15), (9, 3), (9, 9), (9, 15), (15, 3), (15, 9), (15, 15)]
        elif self.board_size == 13:
            star_points = [(3, 3), (3, 9), (6, 6), (9, 3), (9, 9)]
        elif self.board_size == 9:
            star_points = [(2, 2), (2, 6), (4, 4), (6, 2), (6, 6)]
        
        for x, y in star_points:
            pygame.draw.circle(
                self.screen, BLACK,
                (self.margin + x * self.cell_size, self.margin + y * self.cell_size),
                4
            )
        
        # 繪製棋子
        for y in range(self.board_size):
            for x in range(self.board_size):
                stone = self.game.board[x, y]
                if stone != 0:
                    pos = (self.margin + x * self.cell_size, self.margin + y * self.cell_size)
                    if stone == 1:  # 黑子
                        if self.black_stone:
                            self.screen.blit(self.black_stone, (pos[0] - self.cell_size//2, pos[1] - self.cell_size//2))
                        else:
                            pygame.draw.circle(self.screen, BLACK, pos, self.cell_size//2 - 2)
                    else:  # 白子
                        if self.white_stone:
                            self.screen.blit(self.white_stone, (pos[0] - self.cell_size//2, pos[1] - self.cell_size//2))
                        else:
                            pygame.draw.circle(self.screen, WHITE, pos, self.cell_size//2 - 2)
        
        # 標記最後一手
        if self.game.last_move and self.game.last_move != 'pass':
            x, y = self.game.last_move
            pos = (self.margin + x * self.cell_size, self.margin + y * self.cell_size)
            pygame.draw.circle(self.screen, HIGHLIGHT_COLOR, pos, 5, 2)
        
        # 顯示當前玩家
        player_text = "黑方" if self.game.current_player == 1 else "白方"
        text_surface = self.font.render(f"當前玩家: {player_text}", True, BLACK)
        self.screen.blit(text_surface, (10, 10))
        
        # 如果遊戲結束，顯示結果
        if self.game.is_game_over():
            result = self.game.get_winner()
            if result:
                winner, black_score, white_score = result
                if winner == 1:
                    result_text = f"黑方勝! 比分: 黑 {black_score:.1f} - 白 {white_score:.1f}"
                elif winner == -1:
                    result_text = f"白方勝! 比分: 黑 {black_score:.1f} - 白 {white_score:.1f}"
                else:
                    result_text = f"平局! 比分: 黑 {black_score:.1f} - 白 {white_score:.1f}"
                
                text_surface = self.font.render(result_text, True, BLACK)
                text_rect = text_surface.get_rect(center=(self.window_size//2, self.margin//2))
                self.screen.blit(text_surface, text_rect)
        
        # 更新顯示
        pygame.display.flip()
    
    def screen_to_board(self, pos):
        """將屏幕座標轉換為棋盤座標。"""
        x, y = pos
        # 計算到最近的格線交叉點的距離
        board_x = round((x - self.margin) / self.cell_size)
        board_y = round((y - self.margin) / self.cell_size)
        
        # 檢查是否在棋盤範圍內
        if 0 <= board_x < self.board_size and 0 <= board_y < self.board_size:
            return (board_x, board_y)
        return None
    
    def get_human_move(self):
        """獲取人類玩家的移動。"""
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_p:  # 按P鍵虛手
                        return 'pass'
                    elif event.key == pygame.K_q:  # 按Q鍵退出
                        pygame.quit()
                        sys.exit()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # 左鍵點擊
                        pos = pygame.mouse.get_pos()
                        board_pos = self.screen_to_board(pos)
                        if board_pos:
                            return board_pos
            
            # 稍微延遲以減少CPU使用
            time.sleep(0.02)
    
    def play_game(self, ai_player=None, ai_color=-1):
        """進行遊戲。
        
        Args:
            ai_player: AI搜索器（MCTS實例）
            ai_color: AI控制的顏色 (1=黑, -1=白)
        """
        game_over = False
        
        while not game_over:
            # 更新顯示
            self.update_display()
            
            # 如果是AI的回合
            if ai_player and self.game.current_player == ai_color:
                # AI選擇移動
                logger.info("AI思考中...")
                probs, _ = ai_player.get_action_probs(self.game, temperature=0.1)
                move = max(probs.items(), key=lambda x: x[1])[0]
                
                # 延遲一下，以便觀察
                time.sleep(0.5)
                
                # 應用移動
                success = self.game.play(move)
                if not success:
                    logger.warning(f"AI嘗試非法移動: {move}")
                    # 執行虛手作為後備
                    self.game.play('pass')
                
                logger.info(f"AI走子: {move}")
            else:
                # 獲取人類移動
                logger.info("等待玩家落子...")
                move = self.get_human_move()
                
                # 應用移動
                success = self.game.play(move)
                if not success:
                    logger.warning(f"非法移動: {move}")
                    # 提示玩家
                    text_surface = self.font.render("非法移動！請重試。", True, (255, 0, 0))
                    self.screen.blit(text_surface, (10, self.window_size - 30))
                    pygame.display.flip()
                    time.sleep(1)
                else:
                    logger.info(f"玩家走子: {move}")
            
            # 檢查遊戲是否結束
            if self.game.is_game_over():
                self.update_display()
                game_over = True
                logger.info("遊戲結束")
                
                # 等待點擊退出
                waiting = True
                while waiting:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                            waiting = False
                    time.sleep(0.1)
                
                pygame.quit()
                return

def play_console(game, ai_player=None, ai_color=-1):
    """使用控制台界面進行遊戲。"""
    def print_board():
        """顯示當前棋盤。"""
        # ANSI顏色代碼
        BLACK_STONE = '\033[1;30m● \033[0m'
        WHITE_STONE = '\033[1;37m● \033[0m'
        EMPTY = '+ '
        
        # 棋盤坐標
        x_coords = '   ' + ' '.join([chr(97 + i) for i in range(game.size)])
        print(x_coords)
        
        # 棋盤內容
        for y in range(game.size):
            row = f"{game.size - y:2d} "
            for x in range(game.size):
                if game.board[x, y] == 1:
                    row += BLACK_STONE
                elif game.board[x, y] == -1:
                    row += WHITE_STONE
                else:
                    row += EMPTY
            print(row + f" {game.size - y}")
        
        # 再次顯示棋盤坐標
        print(x_coords)
        
        # 顯示當前玩家
        print(f"當前玩家: {'黑方' if game.current_player == 1 else '白方'}")
        
        # 顯示最後一手
        if game.last_move:
            if game.last_move == 'pass':
                print(f"上一手: {'黑方' if game.current_player == -1 else '白方'} 虛手")
            else:
                x, y = game.last_move
                print(f"上一手: {'黑方' if game.current_player == -1 else '白方'} {chr(97 + x)}{game.size - y}")
    
    def get_human_move():
        """從命令行獲取人類玩家的移動。"""
        while True:
            try:
                move_str = input("\n輸入移動 (例如 'e5', 'pass' 虛手, 'quit' 退出): ").strip().lower()
                
                if move_str == 'quit':
                    return 'quit'
                elif move_str == 'pass':
                    return 'pass'
                elif len(move_str) >= 2:
                    x = ord(move_str[0]) - ord('a')
                    y = game.size - int(move_str[1:])
                    
                    if 0 <= x < game.size and 0 <= y < game.size:
                        return (x, y)
                
                print(f"無效的移動: {move_str}")
            except ValueError:
                print(f"無效的移動: {move_str}")
    
    # 遊戲主循環
    game_over = False
    
    while not game_over:
        # 顯示棋盤
        print_board()
        
        # 如果是AI的回合
        if ai_player and game.current_player == ai_color:
            print("\nAI思考中...")
            
            # AI選擇移動
            probs, _ = ai_player.get_action_probs(game, temperature=0.1)
            move = max(probs.items(), key=lambda x: x[1])[0]
            
            # 應用移動
            success = game.play(move)
            if not success:
                print(f"AI嘗試非法移動: {move}")
                # 執行虛手作為後備
                game.play('pass')
            
            if move == 'pass':
                print(f"AI虛手")
            else:
                x, y = move
                print(f"AI走子: {chr(97 + x)}{game.size - y}")
        else:
            # 獲取人類移動
            move = get_human_move()
            
            # 檢查是否退出
            if move == 'quit':
                return
            
            # 應用移動
            success = game.play(move)
            if not success:
                print(f"非法移動: {move}")
        
        # 檢查遊戲是否結束
        if game.is_game_over():
            print_board()
            print("\n遊戲結束")
            
            # 顯示結果
            result = game.get_winner()
            if result:
                winner, black_score, white_score = result
                if winner == 1:
                    print(f"黑方勝! 比分: 黑 {black_score:.1f} - 白 {white_score:.1f}")
                elif winner == -1:
                    print(f"白方勝! 比分: 黑 {black_score:.1f} - 白 {white_score:.1f}")
                else:
                    print(f"平局! 比分: 黑 {black_score:.1f} - 白 {white_score:.1f}")
            
            game_over = True

def parse_args():
    """解析命令行參數。"""
    parser = argparse.ArgumentParser(description='圍棋AI對弈腳本')
    
    # 遊戲模式
    parser.add_argument('--mode', type=str, default='human_vs_ai', 
                      choices=['human_vs_ai', 'ai_vs_ai', 'human_vs_human'],
                      help='遊戲模式')
    
    # 棋盤大小
    parser.add_argument('--size', type=int, default=19, choices=[9, 13, 19],
                      help='棋盤大小')
    
    # 界面選擇
    parser.add_argument('--ui', type=str, default='auto', choices=['gui', 'console', 'auto'],
                      help='界面類型')
    
    # AI相關
    parser.add_argument('--model_path', type=str, required=True,
                      help='AI模型路徑')
    parser.add_argument('--model_type', type=str, default='residual',
                      choices=['residual', 'light', 'cnn_rnn'],
                      help='模型類型')
    parser.add_argument('--mcts_sims', type=int, default=800,
                      help='MCTS模擬次數')
    parser.add_argument('--temperature', type=float, default=0.1,
                      help='MCTS溫度參數')
    
    # 玩家顏色
    parser.add_argument('--color', type=str, default='black',
                      choices=['black', 'white', 'random'],
                      help='人類玩家顏色 (human_vs_ai 模式)')
    
    # 系統相關
    parser.add_argument('--gpu', type=str, default=None,
                      help='指定GPU設備(例如 "0")')
    
    return parser.parse_args()

def set_gpu_options(gpu_str=None):
    """設置GPU選項。"""
    if gpu_str is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_str
        logger.info(f"使用GPU設備: {gpu_str}")
    
    # 設置GPU成長模式
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"發現 {len(gpus)} 個GPU設備，已設置記憶體增長模式")
        except RuntimeError as e:
            logger.error(f"設置GPU記憶體增長模式失敗: {str(e)}")

def load_model(model_path, model_type='residual'):
    """載入AI模型。"""
    logger.info(f"載入模型: {model_path}")
    
    # 確定輸入形狀
    input_shape = (19, 19, 17)  # 標準圍棋特徵平面
    
    # 根據模型類型創建模型
    if model_type == 'residual':
        model = create_go_model(input_shape=input_shape)
    elif model_type == 'light':
        model = create_light_model(input_shape=input_shape)
    elif model_type == 'cnn_rnn':
        model = create_cnn_rnn_model(board_input_shape=input_shape)
    else:
        raise ValueError(f"未知的模型類型: {model_type}")
    
    # 載入權重
    try:
        model.load_weights(model_path)
        logger.info("模型載入成功")
        return model
    except Exception as e:
        logger.error(f"載入模型失敗: {str(e)}")
        return None

def main():
    """主函數。"""
    # 解析命令行參數
    args = parse_args()
    
    # 設置GPU選項
    set_gpu_options(args.gpu)
    
    # 創建遊戲實例
    game = GoGame(size=args.size)
    
    # 決定UI類型
    use_gui = args.ui == 'gui' or (args.ui == 'auto' and HAS_PYGAME)
    
    # 為AI模式載入模型
    ai_model = None
    ai_player = None
    ai_color = None
    
    if args.mode != 'human_vs_human':
        ai_model = load_model(args.model_path, args.model_type)
        if ai_model:
            ai_player = MCTS(
                model=ai_model,
                num_simulations=args.mcts_sims,
                temperature=args.temperature
            )
        else:
            logger.error("無法載入AI模型，回退到human_vs_human模式")
            args.mode = 'human_vs_human'
    
    # 設置玩家顏色
    if args.mode == 'human_vs_ai':
        if args.color == 'black':
            ai_color = -1  # AI執白
        elif args.color == 'white':
            ai_color = 1   # AI執黑
        else:  # random
            ai_color = 1 if np.random.random() < 0.5 else -1
            
        logger.info(f"人類玩家執{'黑' if ai_color == -1 else '白'}")
    
    # 開始遊戲
    if use_gui:
        logger.info("使用圖形界面")
        gui = GoGUI(game)
        
        if args.mode == 'human_vs_ai':
            gui.play_game(ai_player=ai_player, ai_color=ai_color)
        elif args.mode == 'ai_vs_ai':
            gui.play_game(ai_player=ai_player, ai_color=1)  # AI控制雙方
        else:  # human_vs_human
            gui.play_game()
    else:
        logger.info("使用控制台界面")
        
        if args.mode == 'human_vs_ai':
            play_console(game, ai_player=ai_player, ai_color=ai_color)
        elif args.mode == 'ai_vs_ai':
            play_console(game, ai_player=ai_player, ai_color=1)  # AI控制雙方
        else:  # human_vs_human
            play_console(game)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        logger.info("程序被用戶中斷")
    except Exception as e:
        logger.exception(f"程序執行過程中發生錯誤: {str(e)}")
        sys.exit(1)