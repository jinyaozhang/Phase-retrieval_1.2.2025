import tensorflow as tf

class AdamOptimizer:
    def __init__(self, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None  # 一阶矩
        self.v = None  # 二阶矩
        self.t = 0     # 时间步

    def update(self, gradients, parameters):
        # 确保参数是 TensorFlow 的变量
        if not isinstance(parameters, tf.Variable):
            parameters = tf.Variable(parameters)

        # 初始化 m 和 v
        if self.m is None:
            self.m = tf.Variable(tf.zeros_like(parameters, dtype=parameters.dtype), trainable=False)
        if self.v is None:
            self.v = tf.Variable(tf.zeros_like(parameters, dtype=parameters.dtype), trainable=False)

        self.t += 1

        # 更新一阶和二阶矩
        self.m.assign(self.beta1 * self.m + (1 - self.beta1) * gradients)
        self.v.assign(self.beta2 * self.v + (1 - self.beta2) * tf.square(gradients))

        # 偏差校正
        beta1_t = tf.cast(self.beta1 ** self.t, dtype=parameters.dtype)
        beta2_t = tf.cast(self.beta2 ** self.t, dtype=parameters.dtype)
        m_hat = self.m / (1 - beta1_t)
        v_hat = self.v / (1 - beta2_t)

        # 更新参数
        parameters.assign_sub(self.learning_rate * m_hat / (tf.sqrt(v_hat) + self.epsilon))
        return parameters
