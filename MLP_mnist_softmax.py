import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import gzip
import os

def load_mnist_images(filename):
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)  # 跳过前16字节文件头
    return data.reshape(-1, 784).astype(float) / 255.0

def load_mnist_labels(filename):
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=8)   # 跳过前8字节文件头
    return data.astype(int)

# 假设数据在 ./data/MNIST/raw/
data_dir = './data/MNIST/raw'
X_train = load_mnist_images(os.path.join(data_dir, 'train-images-idx3-ubyte.gz'))
y_train = load_mnist_labels(os.path.join(data_dir, 'train-labels-idx1-ubyte.gz'))
X_test = load_mnist_images(os.path.join(data_dir, 't10k-images-idx3-ubyte.gz'))
y_test = load_mnist_labels(os.path.join(data_dir, 't10k-labels-idx1-ubyte.gz'))

from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.15, random_state=42
)

# 将标签转换为 one-hot 编码
def to_onehot(y, num_classes=10):
    onehot = np.zeros((y.shape[0], num_classes))
    onehot[np.arange(y.shape[0]), y] = 1.0
    return onehot

# X_train 为 (42000, 784)，Y_train 为 (42000, 10)
# 输入维度 input_dim = 784

Y_train = to_onehot(y_train)
Y_val = to_onehot(y_val)
Y_test = to_onehot(y_test)

# 两层 MLP 类
class TwoLayerMLP:
    def __init__(self, input_dim, hidden_dim=64, num_classes=10, activation="relu",optimizer="momentum",
                 beta1=0.9, beta2=0.999, init_method="auto", lr=0.1, momentum=0.9, l2_lambda=0.0, seed=0):
        """
        参数:
            input_dim: 输入特征维度
            hidden_dim: 隐藏层神经元个数
            activation: 隐藏层激活函数，支持 "tanh" 或 "relu"
            init_method: 初始化方法，支持 "xavier", "he", "random"，"auto" 会根据激活函数自动选择
            lr: 学习率
            momentum: 动量系数，0 表示没有动量
            l2_lambda: L2 正则化强度，0 表示没有正则化，一般可以是1e-4、1e-3这样
            seed: 随机种子
        """
        self.lr = lr
        self.momentum = momentum
        self.l2_lambda = l2_lambda
        self.activation = activation
        self.num_classes = num_classes  # 多分类新增
        self.optimizer = optimizer
        self.beta1 = beta1 
        self.beta2 = beta2

        rng = np.random.default_rng(seed)

        # 自动选择参数初始化方法
        if init_method == "auto":
            if activation in ["tanh", "sigmoid"]:
                init_method = "xavier"
            elif activation == "relu":
                init_method = "he"
            else:
                init_method = "he"

        # 权重初始化
        if init_method == "xavier":
            # Xavier/Glorot 初始化：适用于 tanh/sigmoid
            limit1 = np.sqrt(6.0 / (input_dim + hidden_dim))
            limit2 = np.sqrt(6.0 / (hidden_dim + num_classes))
            # 约定上下限，并限制维度，初始化w
            self.W1 = rng.uniform(-limit1, limit1, size=(input_dim, hidden_dim))
            self.W2 = rng.uniform(-limit2, limit2, size=(hidden_dim, num_classes))
        elif init_method == "he":
            # He/Kaiming 初始化：适用于 ReLU
            limit1 = np.sqrt(6.0 / input_dim)
            limit2 = np.sqrt(6.0 / hidden_dim)
            self.W1 = rng.uniform(-limit1, limit1, size=(input_dim, hidden_dim))
            self.W2 = rng.uniform(-limit2, limit2, size=(hidden_dim, num_classes))
        else:  # random：简单小随机数（一般不用这个方法）
            self.W1 = rng.normal(0, 0.01, size=(input_dim, hidden_dim))
            self.W2 = rng.normal(0, 0.01, size=(hidden_dim, num_classes))

        # 偏置初始化为零（偏置不影响方差，初始化为零可以更好地限制均值为零）
        self.b1 = np.zeros((1, hidden_dim))
        self.b2 = np.zeros((1, num_classes))

        if optimizer == "momentum":
            self.v_W1 = np.zeros_like(self.W1)
            self.v_b1 = np.zeros_like(self.b1)
            self.v_W2 = np.zeros_like(self.W2)
            self.v_b2 = np.zeros_like(self.b2)
        elif optimizer == "adam":
            self.m_W1 = np.zeros_like(self.W1)
            self.m_b1 = np.zeros_like(self.b1)
            self.m_W2 = np.zeros_like(self.W2)
            self.m_b2 = np.zeros_like(self.b2)
            self.v_W1 = np.zeros_like(self.W1)
            self.v_b1 = np.zeros_like(self.b1)
            self.v_W2 = np.zeros_like(self.W2)
            self.v_b2 = np.zeros_like(self.b2)
            self.t = 0

        # 用于早停的最佳参数（也是正则化的一个方式）
        self.best_params = None
        self.best_val_loss = float("inf")  # 正无穷大

    @staticmethod
    def softmax(z):
         # z: (N, K)
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # 减去最大值防止溢出（很重要）
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    # 激活函数
    @staticmethod  # @staticmethod 是 Python 内置的一个装饰器，用于把类中的方法声明为静态方法。
    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))

    # 选择对应的激活函数
    def hidden_activation(self, z):
        if self.activation == "tanh":
            return np.tanh(z)
        elif self.activation == "relu":
            return np.maximum(0, z)
        else:
            raise ValueError("Unsupported activation")

    # 返回值是梯度
    def hidden_activation_deriv(self, a, z):
        """隐藏层激活函数的导数，a 是激活输出，z 是线性输出"""
        if self.activation == "tanh":
            return 1.0 - a ** 2          # tanh'(z) = 1 - tanh^2(z) = 1 - a^2
        elif self.activation == "relu":
            return (z > 0).astype(float)  # ReLU'(z) = 1 if z>0 else 0
        else:
            raise ValueError("Unsupported activation")

    # 前向传播
    def forward(self, X):
        Z1 = X @ self.W1 + self.b1          # N x h，@ 是矩阵乘法
        A1 = self.hidden_activation(Z1)     # N x h
        Z2 = A1 @ self.W2 + self.b2         # N x k
        A2 = self.softmax(Z2)               # 多分类，这里把sigmoid换成了softmax
        cache = (X, Z1, A1, Z2, A2)         # 保存中间变量供反向传播使用，注意这里面也包含了A2
        return A2, cache

    # 损失函数
    def compute_loss(self, Y, A2):
        eps = 1e-8  # 防止 log(0)
        # 多分类交叉熵
        ce = -np.mean(np.sum(Y * np.log(A2 + eps), axis=1))
        # L2正则化
        if self.l2_lambda > 0:
            l2 = 0.5 * self.l2_lambda * (np.sum(self.W1**2) + np.sum(self.W2**2))
            return ce + l2
        return ce

    # 反向传播（手动链式法则）
    def backward(self, X, Y, cache):
        _, Z1, A1, _, A2 = cache
        N = X.shape[0]

        # 输出层：损失对 Z2 的梯度，注意平均损失带来的 1/N
        dZ2 = (A2 - Y) / N                      # N x k，这个是L对Z2求导的结果，详细的在前置数学知识里面推过了
        dW2 = A1.T @ dZ2                        # h x k，A1是隐藏层激活输出，就是Z2对W2的求导
        db2 = np.sum(dZ2, axis=0, keepdims=True)  # 1 x k

        # 隐藏层，掌握反向传播的数学推导
        dA1 = dZ2 @ self.W2.T                   # N x h
        dZ1 = dA1 * self.hidden_activation_deriv(A1, Z1)  # N x h
        dW1 = X.T @ dZ1                         # d x h
        db1 = np.sum(dZ1, axis=0, keepdims=True)  # 1 x h

        # L2 正则化梯度
        if self.l2_lambda > 0:
            dW2 += self.l2_lambda * self.W2
            dW1 += self.l2_lambda * self.W1

        return dW1, db1, dW2, db2

    # 参数更新
    def update(self, dW1, db1, dW2, db2):
        """
        沿用了make_moons实验中的动量的方法，增加了adam调试器的方法
        """
        if self.optimizer == "momentum":
            # 动量
            self.v_W1 = self.momentum * self.v_W1 + (1 - self.momentum) * dW1
            self.v_b1 = self.momentum * self.v_b1 + (1 - self.momentum) * db1
            self.v_W2 = self.momentum * self.v_W2 + (1 - self.momentum) * dW2
            self.v_b2 = self.momentum * self.v_b2 + (1 - self.momentum) * db2

            self.W1 -= self.lr * self.v_W1
            self.b1 -= self.lr * self.v_b1
            self.W2 -= self.lr * self.v_W2
            self.b2 -= self.lr * self.v_b2

        elif self.optimizer == "adam":
            """
            adam的技巧
            动量：利用梯度的一阶矩（即梯度的指数移动平均）来保持方向惯性
            自适应学习率：利用梯度的二阶矩（即梯度平方的指数移动平均）来为每个参数单独调整学习率
            使得经常更新的参数学习率变小，不常更新的参数学习率变大
            """

            ###
            # 实验发现基于Numpy的手写adam优化器明显训练更慢，单步耗时更多
            # 可能是由于增加了大量的浮点运算
            # 对于手写数字识别这种相对简单的实验，还是用动量更好
            # 并且，用adam之后，相应的学习率和训练轮次等都要适当调整，否则效果很差！！
            # momentum: lr=0.1; epochs=60; lr_decay=0.005
            # adam: lr=0.001; epochs=60; lr_decay=0.0003(甚至更小或无)
            ###

            self.t += 1  # 时间步

            # 一阶矩更新
            self.m_W1 = self.beta1 * self.m_W1 + (1 - self.beta1) * dW1
            self.m_b1 = self.beta1 * self.m_b1 + (1 - self.beta1) * db1
            self.m_W2 = self.beta1 * self.m_W2 + (1 - self.beta1) * dW2
            self.m_b2 = self.beta1 * self.m_b2 + (1 - self.beta1) * db2

            # 二阶矩更新
            self.v_W1 = self.beta2 * self.v_W1 + (1 - self.beta2) * (dW1 ** 2)
            self.v_b1 = self.beta2 * self.v_b1 + (1 - self.beta2) * (db1 ** 2)
            self.v_W2 = self.beta2 * self.v_W2 + (1 - self.beta2) * (dW2 ** 2)
            self.v_b2 = self.beta2 * self.v_b2 + (1 - self.beta2) * (db2 ** 2)

            # 偏差校正
            """
            偏差校正是为了修正初始化时指数移动平均（EMA）估计值偏低的问题。具体来说，Adam 维护两个变量：
            一阶矩 m_t（梯度的指数移动平均）和二阶矩 v_t（梯度平方的指数移动平均）。它们通常被初始化为零，
            """
            m_hat_W1 = self.m_W1 / (1 - self.beta1 ** self.t)
            m_hat_b1 = self.m_b1 / (1 - self.beta1 ** self.t)
            m_hat_W2 = self.m_W2 / (1 - self.beta1 ** self.t)
            m_hat_b2 = self.m_b2 / (1 - self.beta1 ** self.t)

            v_hat_W1 = self.v_W1 / (1 - self.beta2 ** self.t)
            v_hat_b1 = self.v_b1 / (1 - self.beta2 ** self.t)
            v_hat_W2 = self.v_W2 / (1 - self.beta2 ** self.t)
            v_hat_b2 = self.v_b2 / (1 - self.beta2 ** self.t)

            # 参数更新
            eps = 1e-8
            self.W1 -= self.lr * m_hat_W1 / (np.sqrt(v_hat_W1) + eps)
            self.b1 -= self.lr * m_hat_b1 / (np.sqrt(v_hat_b1) + eps)
            self.W2 -= self.lr * m_hat_W2 / (np.sqrt(v_hat_W2) + eps)
            self.b2 -= self.lr * m_hat_b2 / (np.sqrt(v_hat_b2) + eps)

    # 预测
    def predict_proba(self, X):
        A2, _ = self.forward(X)
        return A2

    def predict(self, X):
        A2, _ = self.forward(X)
        return np.argmax(A2, axis=1)

    # 早停相关
    def save_best(self):
        self.best_params = {
            "W1": self.W1.copy(),
            "b1": self.b1.copy(),
            "W2": self.W2.copy(),
            "b2": self.b2.copy(),
        }

    def restore_best(self):
        if self.best_params is not None:
            self.W1 = self.best_params["W1"].copy()
            self.b1 = self.best_params["b1"].copy()
            self.W2 = self.best_params["W2"].copy()
            self.b2 = self.best_params["b2"].copy()


# 数值梯度检查
def numerical_gradient_check(model, X, Y, eps=1e-5):
    """
    对比数值梯度和解析梯度，验证反向传播实现是否正确。
    只应在小网络和小样本上调用。
    """
    params = [model.W1, model.b1, model.W2, model.b2]
    names = ["W1", "b1", "W2", "b2"]

    # 获取解析梯度
    A2, cache = model.forward(X)
    dW1_ana, db1_ana, dW2_ana, db2_ana = model.backward(X, Y, cache)
    ana_grads = [dW1_ana, db1_ana, dW2_ana, db2_ana]

    for param, name, grad_ana in zip(params, names, ana_grads):
        grad_num = np.zeros_like(param)
        # np.nditer 是 NumPy 提供的一个多维数组迭代器，可以高效地逐个访问数组中的元素。它比普通的 for 循环嵌套更简洁，尤其适合处理任意维度的数组
        # 但默认情况下，np.nditer 只返回元素值，不提供索引
        # 加上 flags=["multi_index"] 后，迭代器会跟踪当前元素的多维索引，可以通过 it.multi_index 获取
        it = np.nditer(param, flags=["multi_index"])
        while not it.finished:
            idx = it.multi_index  ###
            old_val = param[idx]

            # L(θ + ε)
            param[idx] = old_val + eps
            loss_plus = model.compute_loss(Y, model.forward(X)[0]) # 这里就是进行一次前向传播，得到A2（对应[0]）,然后再计算损失

            # L(θ - ε)
            param[idx] = old_val - eps
            loss_minus = model.compute_loss(Y, model.forward(X)[0])

            grad_num[idx] = (loss_plus - loss_minus) / (2 * eps)

            # 恢复原值
            param[idx] = old_val
            it.iternext()

        # 相对误差
        numerator = np.linalg.norm(grad_num - grad_ana)  # np.linalg.norm 是 NumPy 中用于计算范数的函数
        denominator = max(np.linalg.norm(grad_num), np.linalg.norm(grad_ana), 1e-12)  # 取较大值，得到相对误差的分母，其中加入一个极小值保证不会除以零
        rel_err = numerator / denominator

        if rel_err > 1e-4:
            print(f"  [!] {name} 梯度可能有问题，请检查反向传播实现。")


# 训练函数
def train(model, X_train, Y_train, y_train=None, X_val=None, Y_val=None, y_val=None,
          epochs=100, batch_size=256, lr_decay=0.0, patience=5, verbose=True):
    """
    训练模型，支持 mini-batch、早停和学习率衰减。
    返回:history: 字典，包含训练损失、验证损失、训练准确率、验证准确率
    """
    N = X_train.shape[0]  # 得到样本数
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    
    no_improve = 0  # 连续未改善的 epoch 数

    for epoch in range(epochs):
        # 打乱训练数据
        idx = np.random.permutation(N)  # 函数返回一个形状为 (N,) 的一维数组，包含从 0 到 N-1 的所有整数，但顺序是随机打乱的
        X_shuffled = X_train[idx]
        Y_shuffled = Y_train[idx]

        # Mini-batch 迭代
        for i in range(0, N, batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            Y_batch = Y_shuffled[i:i+batch_size]

            # 前向 → 反向 → 更新
            A2, cache = model.forward(X_batch)
            dW1, db1, dW2, db2 = model.backward(X_batch, Y_batch, cache)
            model.update(dW1, db1, dW2, db2)

        # 每个 epoch 结束后，在整个训练集上计算损失和准确率（用于监控）
        A2_train, _ = model.forward(X_train)
        train_loss = model.compute_loss(Y_train, A2_train)
        train_acc = np.mean(np.argmax(A2_train, axis=1) == y_train)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)

        # 如果有验证集，计算验证损失和准确率
        if X_val is not None and Y_val is not None:
            A2_val, _ = model.forward(X_val)
            val_loss = model.compute_loss(Y_val, A2_val)
            val_acc = np.mean(np.argmax(A2_val, axis=1) == y_val)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
        else:
            val_loss = train_loss
            val_acc = train_acc
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

        # 学习率衰减（Learning Rate Decay）是一种在训练过程中逐渐降低学习率的策略。
        # 它通常用于训练后期，帮助模型在接近最优解时减小参数更新的步长，从而更稳定地收敛，避免在最优点附近震荡。
        if lr_decay > 0:
            model.lr *= (1 - lr_decay)

        # 早停逻辑
        if val_loss < model.best_val_loss:
            model.best_val_loss = val_loss
            model.save_best()
            no_improve = 0
        else:
            no_improve += 1

        if verbose and epoch % 10 == 0:
            print(f"epoch {epoch + 1:4d} | train_loss: {train_loss:.6f} | train_acc: {train_acc:.4f} | "
                  f"val_loss: {val_loss:.6f} | val_acc: {val_acc:.4f} | lr: {model.lr:.6f}")

        # 早停：连续 patience 个 epoch 验证损失无改善
        if patience > 0 and no_improve >= patience:
            if verbose:
                print(f"Early stopping at epoch {epoch + 1}")
            model.restore_best()
            break

    print(f"epoch {epoch + 1:4d} | train_loss: {train_loss:.6f} | train_acc: {train_acc:.4f} | "
            f"val_loss: {val_loss:.6f} | val_acc: {val_acc:.4f} | lr: {model.lr:.6f}")
    
    # 训练结束后恢复最佳参数
    if model.best_params is not None:
        model.restore_best()

    return history


# 绘图函数
def plot_loss_curve(history):
    """
    损失下降曲线图
    """
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train Loss")
    if "val_loss" in history:  # 防御性判断
        plt.plot(history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curves")
    plt.legend()  # 图例
    plt.grid(True)  # 网格线
    """
    准确率上升曲线图
    """
    plt.subplot(1, 2, 2)
    plt.plot(history["train_acc"], label="Train Acc")
    if "val_acc" in history:
        plt.plot(history["val_acc"], label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curves")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# 主程序
if __name__ == "__main__":
    # 梯度检查（使用小样本）# 多分类中没开启检查
    # check_model = TwoLayerMLP(input_dim=2, hidden_dim=4, activation="tanh", lr=0.1, seed=0)
    # numerical_gradient_check(check_model, X_train[:5], Y_train[:5])

    # 正式训练
    print("\n开始训练")
    model = TwoLayerMLP(
        input_dim=784,
        hidden_dim=128,
        activation="relu",   # 可以换
        optimizer="momentum",
        beta1=0.9,
        beta2=0.999,
        init_method="auto",  # 自动选择 Xavier 或 He
        lr=0.1,
        momentum=0.9,        # 使用动量加速
        l2_lambda=1e-4,      # L2正则
        seed=45
    )

    history = train(
        model, X_train, Y_train, y_train,
        X_val=X_val, Y_val=Y_val, y_val=y_val,
        epochs=50,
        batch_size=64,
        lr_decay=0.03,
        patience=10,
        verbose=True
    )

    # 测试集评估
    test_pred = model.predict(X_test)
    test_acc = np.mean(test_pred == y_test)
    print(f"\nFinal Test Accuracy: {test_acc:.4f}")

    # 绘制损失曲线和决策边界
    plot_loss_curve(history)