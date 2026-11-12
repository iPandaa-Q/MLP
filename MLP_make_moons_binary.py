import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

# 两层 MLP 类
class TwoLayerMLP:
    def __init__(self, input_dim, hidden_dim=16, activation="tanh",
                 init_method="auto", lr=0.1, momentum=0.0, l2_lambda=0.0, seed=0):
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
            limit2 = np.sqrt(6.0 / (hidden_dim + 1))
            # 约定上下限，并限制维度，初始化w
            self.W1 = rng.uniform(-limit1, limit1, size=(input_dim, hidden_dim))
            self.W2 = rng.uniform(-limit2, limit2, size=(hidden_dim, 1))
        elif init_method == "he":
            # He/Kaiming 初始化：适用于 ReLU
            limit1 = np.sqrt(6.0 / input_dim)
            limit2 = np.sqrt(6.0 / hidden_dim)
            self.W1 = rng.uniform(-limit1, limit1, size=(input_dim, hidden_dim))
            self.W2 = rng.uniform(-limit2, limit2, size=(hidden_dim, 1))
        else:  # random：简单小随机数（一般不用这个方法）
            self.W1 = rng.normal(0, 0.01, size=(input_dim, hidden_dim))
            self.W2 = rng.normal(0, 0.01, size=(hidden_dim, 1))

        # 偏置初始化为零（偏置不影响方差，初始化为零可以更好地限制均值为零）
        self.b1 = np.zeros((1, hidden_dim))
        self.b2 = np.zeros((1, 1))

        # 动量速度项（初始化为全零）
        self.v_W1 = np.zeros_like(self.W1)
        self.v_b1 = np.zeros_like(self.b1)
        self.v_W2 = np.zeros_like(self.W2)
        self.v_b2 = np.zeros_like(self.b2)

        # 用于早停的最佳参数（也是正则化的一个方式）
        self.best_params = None
        self.best_val_loss = float("inf")  # 正无穷大

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
        Z2 = A1 @ self.W2 + self.b2         # N x 1
        A2 = self.sigmoid(Z2)               # N x 1
        cache = (X, Z1, A1, Z2, A2)         # 保存中间变量供反向传播使用，注意这里面也包含了A2
        return A2, cache

    # 损失函数
    def compute_loss(self, Y, A2):
        eps = 1e-8  # 防止 log(0)
        # 二分类交叉熵
        bce = -np.mean(Y * np.log(A2 + eps) + (1 - Y) * np.log(1 - A2 + eps))
        # L2正则化
        if self.l2_lambda > 0:
            l2 = 0.5 * self.l2_lambda * (np.sum(self.W1**2) + np.sum(self.W2**2))
            return bce + l2
        return bce

    # 反向传播（手动链式法则）
    def backward(self, X, Y, cache):
        _, Z1, A1, _, A2 = cache
        N = X.shape[0]

        # 输出层：损失对 Z2 的梯度，注意平均损失带来的 1/N
        dZ2 = (A2 - Y) / N                      # N x 1，这个是L对Z2求导的结果，详细的在前置数学知识里面推过了
        dW2 = A1.T @ dZ2                        # h x 1，A1是隐藏层激活输出，就是Z2对W2的求导
        db2 = np.sum(dZ2, axis=0, keepdims=True)  # 1 x 1，np.sum(axis=0) 对 N 个样本求和，结果形状 (1,)，keepdims=True 保持原来的二维形状，变成 (1, 1)，与b2一致。

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

    # 参数更新（带动量）
    def update(self, dW1, db1, dW2, db2):
        # 动量更新：v = beta*v + (1-beta)*g，beta一般取 0.9 左右
        self.v_W1 = self.momentum * self.v_W1 + (1 - self.momentum) * dW1
        self.v_b1 = self.momentum * self.v_b1 + (1 - self.momentum) * db1
        self.v_W2 = self.momentum * self.v_W2 + (1 - self.momentum) * dW2
        self.v_b2 = self.momentum * self.v_b2 + (1 - self.momentum) * db2

        # 参数更新：theta = theta - lr * v
        self.W1 -= self.lr * self.v_W1
        self.b1 -= self.lr * self.v_b1
        self.W2 -= self.lr * self.v_W2
        self.b2 -= self.lr * self.v_b2

    # 预测
    def predict_proba(self, X):
        A2, _ = self.forward(X)
        return A2

    def predict(self, X, threshold=0.5):
        A2, _ = self.forward(X)
        return (A2 >= threshold).astype(int)  # .astype(int) 将布尔值 True/False 转换为整数 1/0

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
    """
    数值梯度检查的目的是验证反向传播的梯度计算是否正确
    在确保正确的情况下可以不需要
    这里保证看懂并理解就写了
    """
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
def train(model, X_train, Y_train, X_val=None, Y_val=None,
          epochs=1000, batch_size=32, lr_decay=0.0, patience=20, verbose=True):
    """
    训练模型，支持 mini-batch、早停和学习率衰减。
    返回：history: 字典，包含训练损失、验证损失、训练准确率、验证准确率
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
        train_acc = np.mean((A2_train >= 0.5).astype(int) == Y_train)  # mean计算平均值就相当于计算准确率
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)

        # 如果有验证集，计算验证损失和准确率
        if X_val is not None and Y_val is not None:
            A2_val, _ = model.forward(X_val)
            val_loss = model.compute_loss(Y_val, A2_val)
            val_acc = np.mean((A2_val >= 0.5).astype(int) == Y_val)
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

        if verbose and epoch % 50 == 0:
            print(f"epoch {epoch + 1:4d} | train_loss: {train_loss:.6f} | train_acc: {train_acc:.4f} | "
                  f"val_loss: {val_loss:.6f} | val_acc: {val_acc:.4f} | lr: {model.lr:.6f}")

        # 早停：连续 patience 个 epoch 验证损失无改善
        if patience > 0 and no_improve >= patience:
            if verbose:
                print(f"Early stopping at epoch {epoch + 1}")
            model.restore_best()
            break

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

# 决策边界
def plot_decision_boundary(model, X, Y, title="Decision Boundary"):
    # 起手一个背景图大小
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    # 接着画网格点
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]
    # 预测这些网格点
    Z = model.predict(grid).reshape(xx.shape)
    # 窗格大小
    plt.figure(figsize=(8, 6))
    # 颜色映射
    # contourf 的全称是 contour fill，即填充等高线，会填充颜色区域
    plt.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.RdYlBu)
    # 画散点图
    plt.scatter(X[:, 0], X[:, 1], c=Y.ravel(), cmap=plt.cm.RdYlBu, edgecolors="k")
    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()


# 主程序
if __name__ == "__main__":
    # 生成数据
    X, y = make_moons(n_samples=1000, noise=0.25, random_state=45)
    # reshape(-1, 1) 将数组的形状变为 (N, 1)，其中 -1 表示“自动推断行数”，1 表示列数为 1
    Y = y.reshape(-1, 1).astype(float)

    # 划分训练/验证/测试
    X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.4, random_state=45)
    X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=45)

    print(f"Train size: {X_train.shape[0]}, Val size: {X_val.shape[0]}, Test size: {X_test.shape[0]}")

    # 梯度检查（使用小样本）
    print("\n数值梯度检查")
    check_model = TwoLayerMLP(input_dim=2, hidden_dim=4, activation="tanh", lr=0.1, seed=0)
    numerical_gradient_check(check_model, X_train[:5], Y_train[:5])

    # 正式训练
    print("\n开始训练")
    model = TwoLayerMLP(
        input_dim=2,
        hidden_dim=32,
        activation="relu",   # 可以换（对于本次实验而言，经过尝试，relu会比tanh好一些）
        init_method="auto",  # 自动选择 Xavier 或 He
        lr=0.1,
        momentum=0.9,        # 使用动量加速
        l2_lambda=0,         # L2正则
        seed=45
    )

    history = train(
        model, X_train, Y_train,
        X_val=X_val, Y_val=Y_val,
        epochs=1000,
        batch_size=32,
        lr_decay=0.001,
        patience=150,
        verbose=True
    )

    # 测试集评估
    test_pred = model.predict(X_test)
    test_acc = np.mean(test_pred == Y_test)
    print(f"\nFinal Test Accuracy: {test_acc:.4f}")

    # 绘制损失曲线和决策边界
    plot_loss_curve(history)
    plot_decision_boundary(model, X_test, Y_test, title=f"Decision Boundary (Test Acc = {test_acc:.4f})")