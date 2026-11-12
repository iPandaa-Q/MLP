import numpy as np

# 不同初始化方式对比
def xavier_init(in_dim, out_dim, distribution="uniform"):
    """Xavier/Glorot 初始化，适用于 tanh/sigmoid"""
    limit = np.sqrt(6.0 / (in_dim + out_dim))
    if distribution == "uniform":
        return np.random.uniform(-limit, limit, size=(in_dim, out_dim))
    else:  # normal
        std = np.sqrt(2.0 / (in_dim + out_dim))
        return np.random.randn(in_dim, out_dim) * std

def he_init(in_dim, out_dim, distribution="uniform"):
    """He/Kaiming 初始化，适用于 ReLU"""
    limit = np.sqrt(6.0 / in_dim)
    if distribution == "uniform":
        return np.random.uniform(-limit, limit, size=(in_dim, out_dim))
    else:
        std = np.sqrt(2.0 / in_dim)
        return np.random.randn(in_dim, out_dim) * std

"""
训练循环

对于每个 epoch（遍历整个训练集一次）:
    打乱训练数据顺序
    将数据分成若干 mini-batch
    对于每个 mini-batch:
        前向传播：计算预测和损失
        反向传播：计算梯度
        参数更新：用优化器更新权重
    记录当前 epoch 的训练损失（和验证损失）
    检查早停条件
"""

