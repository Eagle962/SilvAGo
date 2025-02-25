import numpy as np
from collections import deque
import copy

class GoGame:
    """改進的圍棋遊戲類，負責管理遊戲狀態和規則。"""
    
    def __init__(self, size=19, komi=6.5):
        """初始化圍棋遊戲。
        
        Args:
            size: 棋盤大小，標準為19x19
            komi: 貼目值，標準為6.5
        """
        self.size = size
        self.board = np.zeros((size, size), dtype=int)
        self.current_player = 1  # 1 代表黑棋, -1 代表白棋
        self.ko_point = None  # 打劫禁著點
        self.last_move = None
        self.passes = 0
        self.captured_stones = {1: 0, -1: 0}  # 黑棋: 0, 白棋: 0
        self.komi = komi
        self.history = []  # 棋盤歷史
        self.move_history = []  # 移動歷史
        self.dead_stones = set()  # 標記的死子

    def get_state(self):
        """獲取當前遊戲狀態的深複製，避免引用問題。"""
        state = {
            'board': self.board.copy(),
            'current_player': self.current_player,
            'ko_point': self.ko_point,
            'last_move': self.last_move,
            'passes': self.passes,
            'captured_stones': copy.deepcopy(self.captured_stones),
            'history': [board.copy() for board in self.history],
            'move_history': self.move_history.copy(),
            'dead_stones': copy.deepcopy(self.dead_stones)
        }
        return state
        
    def set_state(self, state):
        """從保存的狀態恢復遊戲。"""
        self.board = state['board'].copy()
        self.current_player = state['current_player']
        self.ko_point = state['ko_point']
        self.last_move = state['last_move']
        self.passes = state['passes']
        self.captured_stones = copy.deepcopy(state['captured_stones'])
        self.history = [board.copy() for board in state['history']]
        self.move_history = state['move_history'].copy()
        self.dead_stones = copy.deepcopy(state['dead_stones'])

    def play(self, move):
        """嘗試在棋盤上落子。返回移動是否合法且成功執行。
        
        Args:
            move: 'pass'表示虛手，或者是(x, y)座標
            
        Returns:
            bool: 移動是否合法且成功執行
        """
        # 保存當前狀態以便回滾
        previous_state = self.get_state()
        
        # 處理虛手
        if move == 'pass':
            self.passes += 1
            self.current_player = -self.current_player
            self.history.append(self.board.copy())
            self.move_history.append('pass')
            self.ko_point = None
            self.last_move = move
            return True

        # 檢查移動的基本合法性
        if not self.is_valid_move(move):
            return False

        x, y = move
        self.board[x, y] = self.current_player
        
        # 處理提子
        captured = self.remove_captured_stones(self.get_opponent())
        
        # 檢查是否自殺，若是則回滾
        if self.is_suicide(x, y) and not captured:
            self.set_state(previous_state)
            return False

        # 處理打劫
        self.handle_ko(x, y, captured)
        
        # 更新狀態
        self.last_move = move
        self.passes = 0
        self.history.append(self.board.copy())
        self.move_history.append(move)
        
        # 檢查是否違反禁全同形
        if self.violates_superko():
            self.set_state(previous_state)
            return False
            
        self.current_player = self.get_opponent()
        return True

    def is_valid_move(self, move):
        """檢查移動是否合法。"""
        if move == 'pass':
            return True
            
        if not isinstance(move, tuple) or len(move) != 2:
            return False
            
        x, y = move
        # 檢查是否在棋盤範圍內
        if not (0 <= x < self.size and 0 <= y < self.size):
            return False
            
        # 檢查位置是否已有棋子
        if self.board[x, y] != 0:
            return False
            
        # 檢查是否是打劫禁點
        if move == self.ko_point:
            return False
            
        return True

    def get_opponent(self):
        """獲取對手顏色。"""
        return -self.current_player

    def get_adjacent_points(self, x, y):
        """獲取相鄰的四個點。"""
        adjacent = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.size and 0 <= ny < self.size:
                adjacent.append((nx, ny))
        return adjacent

    def get_group(self, x, y):
        """使用廣度優先搜索獲取一個連通的棋子群組。"""
        color = self.board[x, y]
        if color == 0:
            return set()
            
        group = set([(x, y)])
        queue = deque([(x, y)])
        
        while queue:
            current = queue.popleft()
            for adj in self.get_adjacent_points(*current):
                if self.board[adj] == color and adj not in group:
                    group.add(adj)
                    queue.append(adj)
                    
        return group

    def get_liberties(self, group):
        """獲取一個群組的氣。"""
        if not group:
            return set()
            
        liberties = set()
        for x, y in group:
            for adj in self.get_adjacent_points(x, y):
                if self.board[adj[0], adj[1]] == 0:
                    liberties.add(adj)
        return liberties

    def remove_captured_stones(self, player_color):
        """移除被吃掉的棋子。返回被吃掉的棋子座標集合。"""
        captured = set()
        for x in range(self.size):
            for y in range(self.size):
                if self.board[x, y] == player_color:
                    group = self.get_group(x, y)
                    if not self.get_liberties(group):
                        for stone in group:
                            self.board[stone] = 0
                            captured.add(stone)
        
        # 更新提子計數
        self.captured_stones[self.current_player] += len(captured)
        return captured

    def is_suicide(self, x, y):
        """檢查最後一手是否自殺。"""
        group = self.get_group(x, y)
        return len(self.get_liberties(group)) == 0

    def handle_ko(self, x, y, captured):
        """處理打劫情況。"""
        # 如果剛好吃掉一顆子，且落子處沒有其他氣
        if len(captured) == 1 and len(self.get_liberties(self.get_group(x, y))) == 1:
            # 設置打劫禁點
            self.ko_point = list(captured)[0]
        else:
            self.ko_point = None

    def violates_superko(self):
        """檢查是否違反禁全同形規則（也稱為超級劫或長劫）。"""
        current_board = self.board.copy()
        # 檢查是否之前出現過相同的棋盤狀態
        for historical_board in self.history[:-1]:  # 排除最後一個（當前棋盤）
            if np.array_equal(current_board, historical_board):
                return True
        return False

    def get_legal_moves(self, include_pass=True):
        """獲取所有合法移動。"""
        moves = []
        if include_pass:
            moves.append('pass')
            
        for x in range(self.size):
            for y in range(self.size):
                move = (x, y)
                if self.is_valid_move(move):
                    # 需要檢查是否自殺或違反禁全同形
                    temp_state = self.get_state()
                    if self.play(move):
                        # 如果合法，還原並添加到合法移動列表
                        self.set_state(temp_state)
                        moves.append(move)
                    else:
                        # 如果不合法，還原棋盤
                        self.set_state(temp_state)
        return moves

    def is_game_over(self):
        """檢查遊戲是否結束（連續兩次虛手）。"""
        return self.passes >= 2

    def mark_dead_stones(self, stones=None):
        """標記死子。可以手動指定或使用啟發式方法自動檢測。"""
        if stones is not None:
            self.dead_stones = set(stones)
            return
            
        # 啟發式自動檢測死子（簡化版）
        self.dead_stones = set()
        
        # 檢查所有棋子群組
        for x in range(self.size):
            for y in range(self.size):
                if self.board[x, y] != 0 and (x, y) not in self.dead_stones:
                    group = self.get_group(x, y)
                    liberties = self.get_liberties(group)
                    
                    # 簡單啟發式：如果氣少於2且被包圍，可能是死子
                    if len(liberties) < 2:
                        is_surrounded = True
                        color = self.board[x, y]
                        opponent = -color
                        
                        # 檢查周圍是否被對手包圍
                        for liberty in liberties:
                            lx, ly = liberty
                            surrounded_liberty = True
                            
                            for adj in self.get_adjacent_points(lx, ly):
                                ax, ay = adj
                                if self.board[ax, ay] != opponent and (ax, ay) not in group:
                                    surrounded_liberty = False
                                    break
                                    
                            if not surrounded_liberty:
                                is_surrounded = False
                                break
                                
                        if is_surrounded:
                            self.dead_stones.update(group)

    def calculate_score(self):
        """計算最終得分。使用區域計分法（中國規則）。"""
        # 先標記死子
        if not self.dead_stones:
            self.mark_dead_stones()
            
        # 計算領地
        territory = self.calculate_territory()
        
        # 計算黑子得分: 黑子數 + 黑方提子數 + 黑方領地
        black_stones = np.sum(self.board == 1)
        black_captured = self.captured_stones[1]
        black_territory = np.sum(territory == 1)
        black_score = black_stones + black_captured + black_territory
        
        # 計算白子得分: 白子數 + 白方提子數 + 白方領地 + 貼目
        white_stones = np.sum(self.board == -1)
        white_captured = self.captured_stones[-1]
        white_territory = np.sum(territory == -1)
        white_score = white_stones + white_captured + white_territory + self.komi
        
        # 處理死子：從擁有方減去，加給對方
        for x, y in self.dead_stones:
            if self.board[x, y] == 1:  # 黑死子
                black_score -= 1
                white_score += 1
            elif self.board[x, y] == -1:  # 白死子
                white_score -= 1
                black_score += 1
        
        return black_score, white_score

    def calculate_territory(self):
        """計算領地。使用改進的洪水填充算法。"""
        territory = np.zeros((self.size, self.size))
        visited = set()

        # 創建棋盤副本，將死子清除
        board_copy = self.board.copy()
        for x, y in self.dead_stones:
            board_copy[x, y] = 0

        def flood_fill(x, y):
            """區域洪水填充算法，標記領地歸屬。"""
            if (x, y) in visited or not (0 <= x < self.size and 0 <= y < self.size):
                return set(), set(), 0
                
            visited.add((x, y))
            
            # 如果是棋子（非死子），返回棋子顏色
            if board_copy[x, y] != 0:
                return set(), set(), board_copy[x, y]
            
            # 空點，進行洪水填充
            empty_points = {(x, y)}
            border_colors = set()
            
            # 遍歷相鄰點
            for nx, ny in self.get_adjacent_points(x, y):
                n_empty, n_border, n_color = flood_fill(nx, ny)
                empty_points |= n_empty
                
                if n_color != 0:
                    border_colors.add(n_color)
            
            # 判斷領地歸屬
            territory_color = 0
            if len(border_colors) == 1:  # 只有一種顏色包圍
                territory_color = border_colors.pop()
                
            return empty_points, border_colors, territory_color

        # 對所有未訪問的空點進行洪水填充
        for x in range(self.size):
            for y in range(self.size):
                if (x, y) not in visited and board_copy[x, y] == 0:
                    empty_points, border_colors, color = flood_fill(x, y)
                    
                    # 如果是單一顏色的領地，標記
                    if color != 0:
                        for ex, ey in empty_points:
                            territory[ex, ey] = color

        return territory

    def get_winner(self):
        """獲取贏家。返回贏家顏色(1為黑，-1為白，0為平局)和雙方得分。"""
        if not self.is_game_over():
            return None
        
        # 計算得分
        black_score, white_score = self.calculate_score()
        
        # 判斷勝負
        if black_score > white_score:
            return 1, black_score, white_score
        elif white_score > black_score:
            return -1, black_score, white_score
        else:
            return 0, black_score, white_score  # 平局
    
    def get_state_features(self, num_history=8):
        """獲取用於神經網絡的棋盤特徵平面。
        
        Args:
            num_history: 歷史棋盤數量，預設為8
            
        Returns:
            特徵平面張量，形狀為(19, 19, 17) - 注意是 channels_last 格式
        """
        # 基本特徵: 當前黑子、白子位置
        features = []
        
        # 當前棋盤狀態
        features.append((self.board == 1).astype(np.float32))  # 黑子
        features.append((self.board == -1).astype(np.float32))  # 白子
        
        # 歷史狀態 (限制特徵總數為17，所以這裡只取少量歷史)
        # 由於我們需要總共17個通道，而已經用了2個當前狀態和1個當前玩家
        # 所以最多只能再加入7對歷史狀態(每對包含黑白兩個通道)
        history_length = min(7, num_history)  # 最多7對歷史狀態
        
        history_boards = []
        if len(self.history) > 0:
            history_boards = self.history[-history_length:]
        
        # 填充歷史記錄
        while len(history_boards) < history_length:
            history_boards.insert(0, np.zeros_like(self.board))
        
        # 添加歷史狀態特徵（最近的7個）
        for i in range(history_length):
            if i < len(history_boards):
                hist_board = history_boards[i]
                features.append((hist_board == 1).astype(np.float32))
                features.append((hist_board == -1).astype(np.float32))
        
        # 當前玩家顏色
        player_feature = np.full((self.size, self.size), self.current_player, dtype=np.float32)
        features.append((player_feature == 1).astype(np.float32))
        
        # 確保特徵數量正確
        while len(features) < 17:
            features.append(np.zeros_like(features[0]))
        
        # 轉換為適合神經網絡的格式 (channels_last)
        features_array = np.array(features)
        
        # 從 (C, H, W) 轉換為 (H, W, C)
        features_array = np.transpose(features_array, (1, 2, 0))
        
        return features_array