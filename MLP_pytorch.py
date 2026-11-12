import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

device = torch.device("cpu")

# 超参数
input_dim = 784        # 28x28 展平
hidden_dim = 128
num_classes = 10
lr = 0.1
momentum = 0.9
batch_size = 64
epochs = 50
patience = 10          # 早停耐心

# 数据预处理：转换为张量并归一化到 [0,1]
transform = transforms.Compose([
    transforms.ToTensor(),                     # 自动将 [0,255] 归一化到 [0,1]
    # transforms.Normalize((0.5,), (0.5,))      # 标准化到 [-1,1],不标准化的效果反而更好
])

# 加载 MNIST
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

# 从训练集中划分验证集
val_size = 10000
train_size = len(train_dataset) - val_size
train_dataset, val_dataset = torch.utils.data.random_split(
    train_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(45)
)

# 创建 DataLoader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 定义 MLP 模型
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(MLP, self).__init__()  # 调用父类的构造函数
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # 全连接层 1
        self.relu = nn.ReLU()  # 激活函数
        self.fc2 = nn.Linear(hidden_dim, num_classes)  # 输出层

    def forward(self, x):
        x = x.view(x.size(0), -1)      # 展平图像 (batch, 1, 28, 28) -> (batch, 784)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)                # 输出 logits，交叉熵损失会自动应用 softmax
        return x

# 实例化模型、损失函数和优化器
model = MLP(input_dim, hidden_dim, num_classes).to(device)
criterion = nn.CrossEntropyLoss()  # 多分类交叉熵，内部含 softmax
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)  # 学习率衰减

# 早停相关
best_val_loss = float('inf')
best_model_state = None
no_improve = 0

# 记录历史
history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

# 训练循环
for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)  # 计算损失
        
        # 反向传播和优化
        optimizer.zero_grad()  # 把梯度置零
        loss.backward()  # 反向传播
        optimizer.step()  # 根据梯度更新参数
        
        # 统计
        train_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()
    
    avg_train_loss = train_loss / train_total
    train_acc = train_correct / train_total
    
    # 验证
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():  # 禁用自动求导
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
    
    avg_val_loss = val_loss / val_total
    val_acc = val_correct / val_total
    
    history['train_loss'].append(avg_train_loss)
    history['val_loss'].append(avg_val_loss)
    history['train_acc'].append(train_acc)
    history['val_acc'].append(val_acc)
    if epoch % 10 == 0:
        print(f"Epoch {epoch + 1:4d} | Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f} | lr: {optimizer.param_groups[0]['lr']:.6f}")

    # 更新学习率
    scheduler.step()
    
    # 早停判断
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_model_state = model.state_dict()
        no_improve = 0
    else:
        no_improve += 1
        if no_improve >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break

# 加载最佳模型并在测试集上评估
model.load_state_dict(best_model_state)
model.eval()
test_correct = 0
test_total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()

test_acc = test_correct / test_total
print(f'Final Test Accuracy: {test_acc:.4f}')

# 绘制损失曲线
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history['train_acc'], label='Train Acc')
plt.plot(history['val_acc'], label='Val Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()