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