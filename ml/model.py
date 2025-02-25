import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

def create_residual_block(x, filters, kernel_size=3):
    """創建一個殘差塊。"""
    shortcut = x
    
    # 第一個卷積層
    x = layers.Conv2D(filters, kernel_size, padding='same', 
                     kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    # 第二個卷積層
    x = layers.Conv2D(filters, kernel_size, padding='same',
                     kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    
    # 殘差連接
    x = layers.Add()([shortcut, x])
    x = layers.ReLU()(x)
    
    return x

def create_go_model(input_shape=(19, 19, 17), num_filters=256, num_res_blocks=19, 
                   include_pass=True, l2_reg=1e-4):
    """創建一個用於圍棋的殘差神經網絡模型。
    
    類似於AlphaZero的架構，使用卷積層和殘差塊。
    
    Args:
        input_shape: 輸入張量的形狀，默認為(19, 19, 17)，代表一個標準的圍棋棋盤
                    和17個特徵平面。
        num_filters: 卷積濾波器的數量
        num_res_blocks: 殘差塊的數量
        include_pass: 是否包含虛手(pass)動作
        l2_reg: L2正則化參數
        
    Returns:
        編譯好的Keras模型
    """
    # 輸入層
    inputs = layers.Input(shape=input_shape)
    
    # 初始卷積層
    x = layers.Conv2D(num_filters, 3, padding='same',
                     kernel_regularizer=regularizers.l2(l2_reg))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    # 殘差塊
    for _ in range(num_res_blocks):
        x = create_residual_block(x, num_filters)
    
    # 策略頭（Policy Head）
    policy_head = layers.Conv2D(2, 1, padding='same',
                              kernel_regularizer=regularizers.l2(l2_reg))(x)
    policy_head = layers.BatchNormalization()(policy_head)
    policy_head = layers.ReLU()(policy_head)
    policy_head = layers.Flatten()(policy_head)
    
    # 如果包含虛手，輸出大小為361+1
    policy_output_size = 19*19 + (1 if include_pass else 0)
    policy_output = layers.Dense(policy_output_size, activation='softmax', 
                                name='policy',
                                kernel_regularizer=regularizers.l2(l2_reg))(policy_head)
    
    # 價值頭（Value Head）
    value_head = layers.Conv2D(1, 1, padding='same',
                             kernel_regularizer=regularizers.l2(l2_reg))(x)
    value_head = layers.BatchNormalization()(value_head)
    value_head = layers.ReLU()(value_head)
    value_head = layers.Flatten()(value_head)
    value_head = layers.Dense(256, activation='relu',
                            kernel_regularizer=regularizers.l2(l2_reg))(value_head)
    value_output = layers.Dense(1, activation='tanh', name='value',
                              kernel_regularizer=regularizers.l2(l2_reg))(value_head)
    
    # 創建模型
    model = models.Model(inputs=inputs, outputs=[policy_output, value_output])
    
    # 編譯模型
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss={
            'policy': 'categorical_crossentropy',
            'value': 'mean_squared_error'
        },
        metrics={
            'policy': 'accuracy',
            'value': 'mean_absolute_error'
        }
    )
    
    return model

def create_light_model(input_shape=(19, 19, 17), num_filters=128, num_res_blocks=10,
                      include_pass=True, l2_reg=1e-4):
    """創建一個更輕量的圍棋模型，適合快速訓練和實驗。"""
    # 與上面的模型相同，但使用更少的濾波器和殘差塊
    return create_go_model(input_shape, num_filters, num_res_blocks, include_pass, l2_reg)

def create_rollout_model(input_shape=(19, 19, 5), output_dim=361):
    """創建一個簡單的快速走子網絡，用於MCTS模擬。"""
    inputs = layers.Input(shape=input_shape)
    
    # 簡單的卷積層
    x = layers.Conv2D(64, 5, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    x = layers.Conv2D(64, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    x = layers.Conv2D(32, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    
    policy_output = layers.Dense(output_dim, activation='softmax', name='policy')(x)
    
    model = models.Model(inputs=inputs, outputs=policy_output)
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# CNN-RNN混合模型，用於利用遊戲序列信息
def create_cnn_rnn_model(board_input_shape=(19, 19, 17), sequence_length=8, 
                        sequence_features=19*19*3, num_filters=128):
    """創建一個CNN-RNN混合模型，結合當前狀態和歷史序列。"""
    # CNN部分 - 處理當前棋盤
    board_input = layers.Input(shape=board_input_shape, name='board_input')
    
    # 初始卷積
    cnn = layers.Conv2D(num_filters, 3, padding='same')(board_input)
    cnn = layers.BatchNormalization()(cnn)
    cnn = layers.ReLU()(cnn)
    
    # 添加殘差塊
    for _ in range(5):
        cnn = create_residual_block(cnn, num_filters)
    
    # RNN部分 - 處理歷史序列
    sequence_input = layers.Input(shape=(sequence_length, sequence_features), 
                                 name='sequence_input')
    
    rnn = layers.LSTM(256, return_sequences=True)(sequence_input)
    rnn = layers.LSTM(256)(rnn)
    
    # 特徵合併
    cnn_flat = layers.Flatten()(cnn)
    combined = layers.Concatenate()([cnn_flat, rnn])
    
    # 策略頭
    policy = layers.Dense(512, activation='relu')(combined)
    policy = layers.Dropout(0.3)(policy)
    policy_output = layers.Dense(361, activation='softmax', name='policy')(policy)
    
    # 價值頭
    value = layers.Dense(256, activation='relu')(combined)
    value = layers.Dropout(0.3)(value)
    value_output = layers.Dense(1, activation='tanh', name='value')(value)
    
    # 創建模型
    model = models.Model(
        inputs=[board_input, sequence_input],
        outputs=[policy_output, value_output]
    )
    
    # 編譯模型
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss={
            'policy': 'categorical_crossentropy',
            'value': 'mean_squared_error'
        },
        metrics={
            'policy': 'accuracy',
            'value': 'mean_absolute_error'
        }
    )
    
    return model