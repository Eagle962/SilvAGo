"""
圍棋AI配置文件
"""

import os

# 基本路徑
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
LOG_DIR = os.path.join(BASE_DIR, 'logs')

# 確保目錄存在
for dir_path in [DATA_DIR, MODEL_DIR, LOG_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# 棋盤配置
BOARD_SIZES = [9, 13, 19]
DEFAULT_BOARD_SIZE = 19
DEFAULT_KOMI = 6.5  # 貼目值

# 模型配置
DEFAULT_MODEL = {
    'name': 'default',
    'type': 'residual',  # 'residual', 'light', 'cnn_rnn'
    'num_filters': 256,
    'num_res_blocks': 19,
    'input_shape': (19, 19, 17),
    'l2_reg': 1e-4,
    'include_pass': True
}

LIGHT_MODEL = {
    'name': 'light',
    'type': 'light',
    'num_filters': 128,
    'num_res_blocks': 10,
    'input_shape': (19, 19, 17),
    'l2_reg': 1e-4,
    'include_pass': True
}

CNN_RNN_MODEL = {
    'name': 'cnn_rnn',
    'type': 'cnn_rnn',
    'board_input_shape': (19, 19, 17),
    'sequence_length': 8,
    'sequence_features': 19*19*3
}

# 訓練配置
TRAINING = {
    'batch_size': 256,
    'epochs_per_iteration': 10,
    'validation_split': 0.1,
    'num_self_play_games': 400,
    'num_mcts_simulations': 800,
    'temperature': 1.0,
    'dirichlet_epsilon': 0.25,
    'dirichlet_alpha': 0.03,
    'c_puct': 5.0,
    'memory_size': 500000,
    'min_buffer_size': 10000,
    'min_train_iterations': 5,
    'lr_schedule': [(0, 0.01), (100000, 0.001), (300000, 0.0001)],
    'momentum': 0.9,
    'weight_decay': 1e-4,
    'checkpoint_freq': 1,
    'use_mixed_precision': False,
    'distribution_strategy': 'mirrored',
    'early_stopping_patience': 5
}

# MCTS配置
MCTS = {
    'num_simulations': 800,
    'c_puct': 5.0,
    'dirichlet_epsilon': 0.25,
    'dirichlet_alpha': 0.03,
    'temperature': 1.0,
    'virtual_loss': 3
}

# 數據處理配置
DATA_PROCESSING = {
    'min_rank': 7,  # 最低段位過濾
    'min_moves': 30,  # 最小移動數
    'hist_planes': 8,  # 歷史特徵平面數
    'train_ratio': 0.9  # 訓練集比例
}

# 評估配置
EVALUATION = {
    'num_games': 100,
    'temperature': 0.1
}

# 遊戲界面配置
GUI = {
    'cell_size': 40,  # 單元格大小（像素）
    'margin': 40  # 邊距（像素）
}

# 顏色配置
COLORS = {
    'BLACK': (0, 0, 0),
    'WHITE': (255, 255, 255),
    'BOARD': (220, 179, 92),
    'GRID': (0, 0, 0),
    'HIGHLIGHT': (255, 0, 0)
}