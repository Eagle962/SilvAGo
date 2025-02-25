import tensorflow as tf
print("TensorFlow 版本:", tf.__version__)
print("是否有可用的 GPU:", tf.config.list_physical_devices('GPU'))
print("GPU 詳細信息:")
print(tf.config.experimental.get_device_details(tf.config.list_physical_devices('GPU')[0]) 
      if tf.config.list_physical_devices('GPU') else "無可用 GPU")

# 簡單測試
with tf.device('/GPU:0'):
    a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    b = tf.constant([[5.0, 6.0], [7.0, 8.0]])
    c = tf.matmul(a, b)
    print(c)