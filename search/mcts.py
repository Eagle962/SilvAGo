import numpy as np
import math
import time
from game.go_game import GoGame

class MCTSNode:
    """蒙特卡洛樹搜索節點類。"""
    
    def __init__(self, game_state=None, parent=None, move=None, prior=0.0):
        """初始化MCTS節點。
        
        Args:
            game_state: 與此節點關聯的遊戲狀態
            parent: 父節點
            move: 到達此節點的移動
            prior: 從策略網絡獲得的先驗概率
        """
        self.game_state = game_state  # 遊戲狀態
        self.parent = parent  # 父節點
        self.move = move  # 到達此節點的動作
        self.children = {}  # 子節點
        
        # MCTS統計信息
        self.visit_count = 0  # 訪問次數
        self.value_sum = 0.0  # 價值總和
        self.prior = prior  # 先驗概率
        
        # 節點是否已完全展開
        self.is_expanded = False
        
    def expanded(self):
        """檢查節點是否已擴展。"""
        return self.is_expanded
        
    def value(self):
        """獲取節點的平均價值。"""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count
    
    def best_child(self, c_puct=1.0):
        """根據UCB公式選擇最佳子節點。
        
        Args:
            c_puct: 探索常數
            
        Returns:
            最佳子節點和對應的移動
        """
        if not self.children:
            return None, None
            
        # 將訪問計數轉化為概率分佈
        counts = np.array([child.visit_count for child in self.children.values()])
        total_count = np.sum(counts)
        
        # 選擇訪問次數最多的子節點
        best_move = max(self.children.items(), 
                        key=lambda item: item[1].visit_count)
        
        return best_move
    
    def select_child(self, c_puct=5.0, dirichlet_noise=False, epsilon=0.25, noise_alpha=0.03):
        """使用PUCT算法選擇子節點。
        
        Args:
            c_puct: 探索常數
            dirichlet_noise: 是否添加Dirichlet噪聲（僅在根節點）
            epsilon: Dirichlet噪聲的權重
            noise_alpha: Dirichlet噪聲的參數
            
        Returns:
            選擇的子節點和對應的移動
        """
        best_score = -float('inf')
        best_move = None
        best_child = None
        
        # 獲取所有子節點的先驗概率
        priors = {move: node.prior for move, node in self.children.items()}
        
        # 如果在根節點，添加Dirichlet噪聲以增加探索性
        if dirichlet_noise:
            noise = np.random.dirichlet([noise_alpha] * len(priors))
            for i, (move, _) in enumerate(priors.items()):
                priors[move] = (1 - epsilon) * priors[move] + epsilon * noise[i]
        
        # 選擇得分最高的子節點
        for move, child in self.children.items():
            # UCB公式：Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
            q_value = child.value() if child.visit_count > 0 else 0.0
            u_value = c_puct * priors[move] * math.sqrt(self.visit_count) / (1 + child.visit_count)
            
            score = q_value + u_value
            
            if score > best_score:
                best_score = score
                best_move = move
                best_child = child
                
        return best_move, best_child


class MCTS:
    """蒙特卡洛樹搜索類。"""
    
    def __init__(self, model, num_simulations=1600, c_puct=5.0, dirichlet_epsilon=0.25,
                dirichlet_alpha=0.03, temperature=1.0, virtual_loss=3):
        """初始化MCTS。
        
        Args:
            model: 神經網絡模型
            num_simulations: 每步模擬次數
            c_puct: 探索常數
            dirichlet_epsilon: Dirichlet噪聲的權重
            dirichlet_alpha: Dirichlet噪聲的參數
            temperature: 溫度參數控制探索/利用
            virtual_loss: 虛擬損失參數
        """
        self.model = model
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.dirichlet_epsilon = dirichlet_epsilon
        self.dirichlet_alpha = dirichlet_alpha
        self.temperature = temperature
        self.virtual_loss = virtual_loss
        
        # 跟蹤已訪問過的狀態以避免重複計算
        self.evaluated_states = {}
    
    def get_action_probs(self, game, temperature=None):
        """執行MCTS並返回移動概率分佈。
        
        Args:
            game: 當前的遊戲實例
            temperature: 溫度參數，控制探索/利用的平衡
            
        Returns:
            移動字典和概率
        """
        if temperature is None:
            temperature = self.temperature
            
        # 創建根節點
        root = MCTSNode(game_state=game.get_state())
        
        # 執行指定次數的模擬
        for i in range(self.num_simulations):
            node = root
            sim_game = GoGame()
            sim_game.set_state(game.get_state())
            
            # 1. 選擇: 從根節點向下選擇最佳節點直到達到葉節點
            while node.expanded() and not sim_game.is_game_over():
                # 使用PUCT算法選擇子節點
                is_root = (node == root)
                move, node = node.select_child(
                    self.c_puct, 
                    dirichlet_noise=is_root,
                    epsilon=self.dirichlet_epsilon, 
                    noise_alpha=self.dirichlet_alpha
                )
                
                if move is None:
                    break
                    
                # 應用移動到模擬遊戲
                sim_game.play(move)
            
            # 2. 擴展: 如果節點未展開且遊戲未結束，使用神經網絡評估狀態
            if not sim_game.is_game_over() and not node.expanded():
                # 獲取棋盤特徵
                features = self._get_features(sim_game)
                
                # 查找緩存或使用網絡評估
                state_key = self._get_state_key(sim_game)
                if state_key in self.evaluated_states:
                    policy, value = self.evaluated_states[state_key]
                else:
                    policy, value = self._evaluate_state(features)
                    self.evaluated_states[state_key] = (policy, value)
                
                # 獲取合法移動
                legal_moves = sim_game.get_legal_moves()
                
                # 創建子節點
                for move in legal_moves:
                    # 獲取移動的先驗概率
                    if move == 'pass':
                        prior = policy.get('pass', 0.01)  # 如果網絡支持pass動作
                    else:
                        x, y = move
                        move_idx = x * sim_game.size + y
                        prior = policy.get(move_idx, 0.0)
                    
                    # 添加子節點
                    node.children[move] = MCTSNode(
                        parent=node,
                        move=move,
                        prior=prior
                    )
                
                # 標記節點為已展開
                node.is_expanded = True
                
                # 更新節點的價值
                node.value_sum += value
                node.visit_count += 1
            else:
                # 遊戲結束，使用實際結果
                if sim_game.is_game_over():
                    outcome = sim_game.get_winner()[0]
                    # 從當前玩家的角度轉換結果
                    value = outcome * sim_game.current_player
                else:
                    # 使用網絡評估
                    features = self._get_features(sim_game)
                    _, value = self._evaluate_state(features)
                
                # 3. 反向傳播: 更新節點統計信息
                while node is not None:
                    node.value_sum += value
                    node.visit_count += 1
                    # 從父節點的角度翻轉價值
                    value = -value
                    node = node.parent
        
        # 根據溫度參數獲取訪問計數
        counts = {move: node.visit_count for move, node in root.children.items()}
        total_count = sum(counts.values())
        
        # 使用溫度參數計算概率
        if temperature == 0:  # 確定性選擇最佳移動
            best_move = max(counts.items(), key=lambda x: x[1])[0]
            probs = {move: 1.0 if move == best_move else 0.0 for move in counts}
        else:  # 根據計數的指數和溫度計算概率
            counts_powered = {move: count ** (1.0 / temperature) for move, count in counts.items()}
            total_powered = sum(counts_powered.values())
            probs = {move: count / total_powered for move, count in counts_powered.items()}
        
        return probs, root
    
    def _get_features(self, game):
        """獲取遊戲狀態的特徵表示。"""
        return game.get_state_features()
    
    def _evaluate_state(self, features):
        """使用神經網絡評估棋盤狀態。
        
        Args:
            features: 棋盤特徵，形狀為(19, 19, 17)
            
        Returns:
            (policy, value): 政策（移動概率）和狀態價值
        """
        # 準備網絡輸入，確保維度正確
        X = np.expand_dims(features, axis=0)  # 添加批次維度
        
        # 使用模型預測
        policy_output, value_output = self.model.predict(X, verbose=0)
        
        # 處理政策輸出（轉為字典）
        policy = {}
        for i in range(361):  # 19x19棋盤
            policy[i] = policy_output[0][i]
        
        # 如果模型支持虛手，添加pass概率
        if policy_output.shape[1] > 361:
            policy['pass'] = policy_output[0][361]
        
        # 獲取價值（範圍-1到1）
        value = value_output[0][0]
        
        return policy, value
    
    def _get_state_key(self, game):
        """為遊戲狀態創建一個唯一的哈希鍵。"""
        # 使用棋盤和當前玩家作為狀態鍵
        return hash(str(game.board.tobytes()) + str(game.current_player))