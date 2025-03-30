import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.distributions import Categorical

# 定义策略网络，用于选择教师网络
class PPOPolicyNetwork(nn.Module):
    def __init__(self, input_size, num_teachers):
        super(PPOPolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, num_teachers)

    def forward(self, x, tau=1.0):
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)
        teacher_probs = F.gumbel_softmax(logits, tau=tau, hard=True)  # 使用Gumbel-Softmax进行选择
        return teacher_probs

# 定义价值网络，用于估算选择的价值
class ValueNetwork(nn.Module):
    def __init__(self, input_size):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        value = self.fc2(x)
        return value

# 知识蒸馏函数，通过选择的教师网络对学生网络进行训练
def distill(teacher_models, student_model, data, teacher_choice):
    distillation_losses = []
    for i in range(data.size(0)):
        selected_teacher = teacher_models[teacher_choice[i]]
        teacher_output = selected_teacher(data[i].unsqueeze(0))
        student_output = student_model(data[i].unsqueeze(0))
        distillation_loss = F.kl_div(
            F.log_softmax(student_output, dim=1),
            F.softmax(teacher_output, dim=1),
            reduction='batchmean'
        )
        distillation_losses.append(distillation_loss)
    return torch.stack(distillation_losses).mean()

# PPO Agent
class PPOAgent:
    def __init__(self, policy_net, value_net, optimizer, gamma=0.99, clip_epsilon=0.2):
        # 初始化 PPO agent，传入策略网络、价值网络、优化器、折扣因子和裁剪范围
        self.policy_net = policy_net    # 策略网络，用于选择教师
        self.value_net = value_net      # 价值网络，用于估算每个状态-动作对的价值
        self.optimizer = optimizer      # 优化器，用于优化策略网络和价值网络
        self.gamma = gamma              # 折扣因子，控制未来奖励在当前价值中的权重
        self.clip_epsilon = clip_epsilon  # PPO算法中用于裁剪比率的范围，防止更新过大

    def update(self, trajectories):
        # 更新策略和价值网络，使用传入的多条轨迹数据
        for trajectory in trajectories:
            # 解包轨迹中的状态、动作、奖励、旧的log概率和价值
            states, actions, rewards, log_probs, values = trajectory

            # 计算优势函数（advantage），即真实的奖励减去价值网络估算的价值
            advantages = rewards - values

            # 重新计算新的log概率，用于和旧log概率比对
            new_log_probs = self.policy_net(states)  # 新的log概率，基于当前策略网络重新计算

            # 计算旧的和新的log概率的比率，用于PPO的目标函数
            ratios = torch.exp(new_log_probs - log_probs)  # log概率的指数差得到概率比率

            # 计算两种策略损失：直接乘以优势的损失，以及带有裁剪的损失
            surr1 = ratios * advantages   # 未裁剪的策略损失
            surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages  # 裁剪策略损失

            # 计算策略损失和价值损失
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(value_net(states), rewards)

            # 更新策略网络
            policy_optimizer.zero_grad()
            policy_loss.backward()
            policy_optimizer.step()

            # 更新价值网络
            value_optimizer.zero_grad()
            value_loss.backward()
            value_optimizer.step()

            # 使用PPO的损失函数，即两种损失的最小值
            policy_loss = -torch.min(surr1, surr2).mean()   # 取最小值并求均值，以保证稳定更新

            # 计算价值网络的损失，用均方误差衡量估计的价值与实际奖励的差距
            value_loss = F.mse_loss(self.value_net(states), rewards)

            # 梯度更新：将策略损失和价值损失加总，反向传播并更新参数
            self.optimizer.zero_grad()         # 清除之前的梯度
            (policy_loss + value_loss).backward()  # 计算总损失的梯度
            self.optimizer.step()              # 更新策略和价值网络的参数


# 计算分类准确率
def compute_batch_accuracy(student_model, data, targets):
    student_output = student_model(data)
    predicted = torch.argmax(student_output, dim=1)
    correct = (predicted == targets).sum().item()
    accuracy = correct / len(targets)
    return accuracy

# CIFAR-10 数据集加载
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# 初始化教师网络，学生网络，策略网络和价值网络
teacher_models = [nn.Sequential(nn.Flatten(), nn.Linear(32 * 32 * 3, 10)) for _ in range(4)]
student_model = nn.Sequential(nn.Flatten(), nn.Linear(32 * 32 * 3, 10))
policy_net = PPOPolicyNetwork(input_size=32 * 32 * 3, num_teachers=4)
value_net = ValueNetwork(input_size=32 * 32 * 3)
optimizer = optim.Adam(list(policy_net.parameters()) + list(value_net.parameters()), lr=0.001)
agent = PPOAgent(policy_net, value_net, optimizer)

# 训练循环
for epoch in range(10):
    trajectories = []
    for data, targets in train_loader:
        # 选择教师网络
        teacher_probs = policy_net(data)
        teacher_choices = torch.argmax(teacher_probs, dim=1)

        # 计算蒸馏损失
        distillation_loss = distill(teacher_models, student_model, data, teacher_choices)

        # 计算奖励
        accuracy_before = compute_batch_accuracy(student_model, data, targets)
        optimizer.zero_grad()
        distillation_loss.backward()
        optimizer.step()
        accuracy_after = compute_batch_accuracy(student_model, data, targets)
        reward = accuracy_after - accuracy_before

        # 记录 PPO 所需的轨迹
        trajectories.append((data, teacher_choices, reward, teacher_probs.log(), value_net(data), epoch))

    # PPO 更新
    agent.update(trajectories)

# 保存模型
torch.save(policy_net.state_dict(), 'policy_network.pth')
torch.save(student_model.state_dict(), 'student_model.pth')
