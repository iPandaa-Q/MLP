# MLP 学习与实验

从数学推导 → NumPy 手写前向/反向传播 → 梯度检查 → 优化器与正则化 → PyTorch 框架重写的完整 MLP 学习链路。

## 仓库结构

| 文件 | 内容 |
| --- | --- |
| `MLP_make_moons_binary.py` | 实验 1：NumPy 手写两层 MLP 二分类（make_moons），含数值梯度检查、早停、学习率衰减、L2 正则化、决策边界可视化 |
| `MLP_mnist_softmax.py` | 实验 2/3：NumPy 手写 MLP 多分类（MNIST），softmax + one-hot，支持动量 SGD / Adam 两种优化器 |
| `MLP_pytorch.py` | 实验 4：PyTorch 重写 MLP（MNIST 多分类），nn.Module + DataLoader + 早停保存 state_dict |
| `MLP_moremethon.py` | make_moons 超参数网格搜索（隐藏层宽度、激活函数、学习率、动量） |
| `MLP_test.py` | Xavier / He 初始化实现与训练循环说明 |

## 运行方式

```bash
# 实验 1：二分类（训练结束后弹出损失曲线和决策边界图）
python MLP_make_moons_binary.py

# 实验 2：手写多分类（需要 ./data/MNIST/raw 下的 4 个 .gz 文件，见数据说明）
python MLP_mnist_softmax.py

# 实验 4：PyTorch 版（torchvision 自动下载 MNIST 到 ./data）
python MLP_pytorch.py

# 超参数网格搜索（实验 1 的扩展）
python MLP_moremethon.py
```

## 数据说明

`data/` 未纳入版本控制（MNIST 原始文件体积较大）。两种获取方式：

- 先运行 `python MLP_pytorch.py`（`download=True`），torchvision 会把 4 个 `.gz` 文件下载到 `./data/MNIST/raw/`，手写版即可直接使用；
- 或从 [MNIST 官网](http://yann.lecun.com/exdb/mnist/) 手动下载，放到 `./data/MNIST/raw/`。

## 一、学习内容

### 两层 MLP 数学结构

- 前向传播：线性变换 + 激活函数 + 输出层。
- 二分类：输出层 sigmoid + 二分类交叉熵。
- 多分类：输出层 softmax + 多分类交叉熵。
- 损失采用平均形式，反向传播中梯度带有 1/N 因子。

### 反向传播

- 手动推导链式法则，完成输出层、隐藏层、输入层参数梯度。
- 实现 `dZ2 = (A2 - Y) / N`、`dW2 = A1.T @ dZ2`、`dA1 = dZ2 @ W2.T` 等核心步骤。
- 使用中心差分进行数值梯度检查，验证解析梯度正确性。

### 训练与优化

- 参数初始化：Xavier、He。
- 激活函数：tanh、ReLU。
- 优化器：SGD、动量 SGD、Adam。
- 训练技巧：mini-batch、数据打乱、早停、学习率衰减、L2 正则化。
- 理解激活饱和、梯度消失、学习率过大导致震荡或停滞等问题。

### 多分类扩展

- one-hot 编码。
- softmax 数值稳定实现：减去每行最大值。
- 预测使用 argmax。

### PyTorch 入门

- Tensor、自动求导、nn.Module、nn.Linear、nn.ReLU、nn.CrossEntropyLoss、DataLoader、optim.SGD。
- 训练循环：`zero_grad()`、`backward()`、`step()`。
- 验证使用 `model.eval()` 和 `torch.no_grad()`。
- 早停保存最佳参数：`state_dict()` 与 `load_state_dict()`。

## 二、实验过程与结果

### 实验 1：手写 MLP 二分类（make_moons）

- 网络结构：输入 2 维 → 隐藏层 32 或 64 个神经元 → 输出 1 维。
- 激活函数：tanh / ReLU。
- 优化器：动量 SGD，学习率 0.1，动量 0.9。
- 使用早停、学习率衰减实验。
- 数值梯度检查通过（相对误差约 1e-8 量级）。
- 最终测试准确率：约 0.955。
- 决策边界能够呈现弯曲形状。

### 实验 2：手写 MLP 多分类（MNIST 0~9）

- 网络结构：输入 784 维 → 隐藏层 128 个神经元 → 输出 10 类。
- 激活函数：ReLU。
- 初始化：He 初始化。
- 优化器：动量 SGD，学习率 0.1，动量 0.9。
- 损失：多分类交叉熵。
- 标签：one-hot 编码。
- 早停 patience=10，训练约 21 个 epoch 后早停。
- 最终测试准确率：0.9792。

### 实验 3：Adam 优化器实现

- 手动实现一阶矩、二阶矩、偏差校正。
- 理解 Adam 常用学习率 0.001。
- 在简单任务上手写 Adam 单步计算量较大，收敛速度不一定优于动量 SGD。

### 实验 4：PyTorch 重写 MLP（MNIST 多分类）

- 使用 `nn.Linear(784,128)` → ReLU → `nn.Linear(128,10)`。
- 损失函数：`nn.CrossEntropyLoss`。
- 优化器：`optim.SGD`，学习率 0.1，动量 0.9。
- 数据预处理中额外使用了 `Normalize((0.5,), (0.5,))`。
- 训练约 21 个 epoch 后早停。
- 最终测试准确率：0.9669。
- 分析：准确率略低于手写版，主要原因是额外标准化改变了输入分布，且训练轮数较少；去掉 Normalize、仅保留 `ToTensor()` 后应更接近手写版结果。

## 三、总体结果

| 实验 | 数据集 | 实现方式 | 测试准确率 |
| --- | --- | --- | --- |
| 二分类 MLP | make_moons | NumPy 手写 | 约 0.955 |
| 多分类 MLP | MNIST 0~9 | NumPy 手写 | 0.9792 |
| 多分类 MLP | MNIST 0~9 | PyTorch | 0.9669 |

> 注：表格中 PyTorch 版 0.9669 对应使用 `Normalize((0.5,), (0.5,))` 时的结果；当前 `MLP_pytorch.py` 已按上述分析将该行注释掉（仅保留 `ToTensor()`），重新运行的结果会更接近手写版。

通过以上实验，完成了从数学推导、手写前向/反向传播、梯度检查、优化器与正则化，到 PyTorch 框架重写的完整 MLP 学习链路。
