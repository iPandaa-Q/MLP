import numpy as np
import itertools
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from MLP_make_moons_binary import TwoLayerMLP
from MLP_make_moons_binary import train

# 生成数据
X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)
Y = y.reshape(-1, 1).astype(float)
X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.4, random_state=42)
X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42)

# 定义超参数网格
hidden_dims = [8, 16, 32, 64]
activations = ["tanh", "relu"]
lrs = [0.01, 0.05, 0.1, 0.2]
momentums = [0.0, 0.9]

results = []

# 遍历所有组合
for h, act, lr, mom in itertools.product(hidden_dims, activations, lrs, momentums):
    # 创建模型（init_method="auto" 会根据激活函数自动选择）
    model = TwoLayerMLP(
        input_dim=2,
        hidden_dim=h,
        activation=act,
        init_method="auto",
        lr=lr,
        momentum=mom,
        l2_lambda=0.0,
        seed=0
    )
    
    # 训练（关闭详细输出）
    train(model, X_train, Y_train, X_val, Y_val,
          epochs=500, batch_size=32, patience=20, verbose=False)
    
    # 验证准确率
    val_acc = np.mean(model.predict(X_val) == Y_val)
    results.append((h, act, lr, mom, val_acc))
    print(f"h={h:2d}, act={act:5s}, lr={lr:.2f}, mom={mom:.1f} -> val_acc={val_acc:.4f}")

# 按验证准确率降序排列
results.sort(key=lambda x: x[4], reverse=True)

print("\n===== Top 10 超参数组合 =====")
for i, (h, act, lr, mom, acc) in enumerate(results[:10], 1):
    print(f"{i:2d}. h={h:2d}, act={act:5s}, lr={lr:.2f}, mom={mom:.1f} -> val_acc={acc:.4f}")