import torch
import torch.optim as optim
from models.h2gcn import H2GCN
from transformers import AutoTokenizer as LLMTokenizer
from transformers import LlamaForCausalLM as LLMModel
from transformers import LlamaConfig as LLMConfig
import numpy as np
import random

from torch.distributions import Categorical
from config import LEARNING_RATE, EPOCHS, SAVE_STEPS, VAL_SET_SIZE, TARGET_MODULES_policy, TARGET_MODULES_value
from config import MICRO_BATCH_SIZE, BATCH_SIZE, GRADIENT_ACCUMULATION_STEPS
from config import PATH_MODEL_PRETRAIN_policy, PATH_MODEL_PRETRAIN_value, DATA_PATH, MODEL_SAVE_DIR, REPO_ID_policy, REPO_ID_value
from config import IS_PARALLELIZABLE, MODEL_PARALLEL, USE_CACHE
from config import MAX_LENGTH_Q, MAX_LENGTH_A, MAX_LENGTH_QA
from config import LORA_DROPOUT, LORA_ALPHA, LORA_R
from config import USE_CUDA, WEIGHT_DECAY
from config import USE_ALL_LOSS
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from model_merge import MLP_P, MLP_V, MergeModel_P, MergeModel_V
from read import get_raw_text_webkb
from src.data_new import get_data
from get_prompts import prompt_collections
from torch.cuda.amp import autocast, GradScaler

def eidx_to_sp(n: int, edge_index: torch.Tensor, device=None) -> torch.sparse.Tensor:
    indices = edge_index
    values = torch.FloatTensor([1.0] * len(edge_index[0])).to(edge_index.device)
    coo = torch.sparse_coo_tensor(indices=indices, values=values, size=[n, n])
    if device is None:
        device = edge_index.device
    return coo.to(device)


def kl_divergence(p, q):
    p = p + 1e-7
    q = q + 1e-7

    # 计算 KL 散度
    kl_loss = torch.sum(p * torch.log(p / q), dim=-1).mean()
    return kl_loss


def distill(node_index, logits, student, features, edge_index, teacher_assignment, labels, optimizer_s, e, mask):
    # 计算初始学生模型的 logits
    student_logits, _, _ = student(eidx_to_sp(len(features), edge_index.detach().cpu()).to(device), features)
    student_logit = student_logits[node_index]

    # 提取指定教师网络的 logits
    teacher_logits = logits[:, node_index, :]
    selected_logit = teacher_assignment @ teacher_logits

    for epoch in range(100):
        student.train()
        optimizer_s.zero_grad()  # 每次迭代前清空梯度

        # 计算新的学生模型 logits
        student_logits, _, _ = student(eidx_to_sp(len(features), edge_index.detach().cpu()).to(device), features)
        ce_loss = F.nll_loss(student_logits[mask], labels[mask])
        student_logit = student_logits[node_index]

        # 计算 distill_loss 和 ent_loss
        distill_loss = kl_divergence(selected_logit, student_logit)
        ent_loss = (-student_logit * torch.log(student_logit + 1e-15)).sum(dim=-1).mean()

        # 计算总损失并反向传播
        total_loss = 0.5 * distill_loss + 0.1 * ent_loss + ce_loss
        total_loss.backward()  # 不使用 retain_graph=True

        # 更新学生模型参数
        optimizer_s.step()
        if epoch % 50 == 0:
            print("Node Index: [{}/{}] Epoch: [{}/{}/{}] Student Model training: distill loss: {:.4f}, ent loss: {:.4f}".format(node_index, len(features), e, 300, epoch, distill_loss.item(),
                                                                                          ent_loss.item()))

    # 在 no_grad 模式下评估学生模型
    with torch.no_grad():
        student_logits, pred, _ = student(eidx_to_sp(len(features), edge_index.detach().cpu()).to(device), features)
        # kl_diff = -kl_divergence(selected_logit, student_logits[node_index])
        ce_diff = F.nll_loss(student_logits[mask], labels[mask])
        acc = compute_accuracy_teacher(pred[mask], labels[mask])

        # 计算最终的 distill diff
        # diff = acc - 0.5 * kl_diff - 0.5 * ce_diff
        diff = acc - ce_diff
    return diff

# def critic(logits, teacher_assignment, labels):
#     selected_logit = teacher_assignment @ teacher_logits

def compute_accuracy(student_model, feature, edge_index, label):
    _, prediction = student_model(feature, edge_index)
    correct = (prediction == label).sum().item()
    accuracy = correct / feature.size(0) * 100
    return accuracy

def compute_accuracy_teacher(prediction, label):
    # _, prediction = student_model(feature, edge_index)
    correct = (prediction == label).sum().item()
    accuracy = correct / label.size(0) * 100
    return accuracy

class PPOAgent:
    def __init__(self, model_policy, model_value, optimizer_1, optimizer_2, gamma=0.99, clip_epsilon=0.2):
        self.policy_net = model_policy
        self.value_net = model_value
        self.optimizer_policy = optimizer_1
        self.optimizer_value = optimizer_2
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon

    def take_action(self, state):
        action_probs = self.policy_net(**state)
        dist = Categorical(action_probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def compute_advantage(self, rewards, values):
        # 计算优势函数
        advantages = []
        cumulative_return = 0
        for reward, value in zip(reversed(rewards), reversed(values)):
            cumulative_return = reward + self.gamma * cumulative_return
            advantage = cumulative_return - value
            advantages.insert(0, advantage)
        return torch.tensor(advantages).cuda()

    def update(self, trajectories):
        # torch.autograd.set_detect_anomaly(True)
        # scaler_2 = GradScaler()
        for trajectory in trajectories:
            states_policy, states_value, actions, log_probs, rewards, values, epoch = trajectory
            rewards_r = torch.tensor(rewards, dtype=torch.float32, requires_grad=True).cuda()

            # 计算优势函数
            advantages = rewards - values
            advantages_r = torch.tensor(advantages, dtype=torch.float32, requires_grad=True).cuda()

            # 更新策略网络
            with autocast():
                probs = self.policy_net(**states_policy)
                # probs = F.softmax(logits, dim=-1)
                distribution = Categorical(probs)
                new_log_probs = distribution.log_prob(torch.tensor(action).to(device))
                # actions, log_probs = self.policy_net(**states_policy)  # 重新计算 log_probs
                ratios = torch.exp(new_log_probs - log_probs.detach())
                surr1 = ratios * advantages_r
                surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages_r
                policy_loss = -torch.min(surr1, surr2).mean()

                # 更新价值网络
                predicted_values = self.value_net(**states_value)
                value_loss = F.mse_loss(predicted_values, rewards_r)

            # 梯度更新
            # scaler_2.scale(policy_loss).backward()
            # scaler_2.scale(value_loss).backward()
            #
            # scaler_2.step(self.optimizer_policy)
            # scaler_2.step(self.optimizer_value)
            # scaler_2.update()
            self.optimizer_policy.zero_grad()
            policy_loss.backward()
            self.optimizer_policy.step()
            self.optimizer_value.zero_grad()
            value_loss.backward()
            self.optimizer_value.step()
            print("Epoch: {} Policy Loss: {} Value Loss: {} Value: {} Reward: {}".format(epoch, policy_loss.item(),
                                                                                         value_loss.item(),
                                                                                         predicted_values.item(),
                                                                                         rewards_r))

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
label_rate = 1
dataset_name = 'texas'
data, _ = get_raw_text_webkb(dataset=dataset_name, use_text=True, seed=0)
features = torch.tensor(np.load('{}_feature.npy'.format(dataset_name))).cuda()
edge_index = torch.tensor(np.load('{}_edge_index.npy'.format(dataset_name))).cuda()
# labels = torch.tensor(np.load('{}_labels.npy'.format(dataset_name))).cuda()
indices = torch.tensor(np.load('filtered_index_{}.npy'.format(dataset_name))).cuda()
labels = torch.tensor(np.load('filtered_padded_predict_{}.npy'.format(dataset_name))).cuda()
labels_true= torch.tensor(np.load('{}_labels.npy'.format(dataset_name))).cuda()
mask = torch.tensor(np.array([item in list(indices) for item in np.arange(len(features))]))

tokenizer_policy = LLMTokenizer.from_pretrained(PATH_MODEL_PRETRAIN_policy, add_eos_token=True, trust_remote_code=True)
ID_START = 128006
ID_END = 128007
ID_BOS = 128000
ID_EOS = 128009
ID_PAD = ID_EOS
ID_BR = 198  # "\n"
ID_SYSTEM = 9125
ID_MODEL = 78191
ID_USER = 882
tokenizer_policy.pad_token_id = ID_EOS
tokenizer_policy.eos_token_id = ID_EOS
tokenizer_policy.padding_side = "right"  # NO use attention-mask
tokenizer_value = LLMTokenizer.from_pretrained(PATH_MODEL_PRETRAIN_value, add_eos_token=True, trust_remote_code=True)
tokenizer_value.pad_token_id = ID_EOS
tokenizer_value.eos_token_id = ID_EOS
tokenizer_value.padding_side = "right"  # NO use attention-mask
out_dim_policy = 128256
out_dim_value = 128256
sequence_length_policy = 128
sequence_length_value = 128
llm_policy = LLMModel.from_pretrained(PATH_MODEL_PRETRAIN_policy, torch_dtype=torch.bfloat16)
llm_policy.is_parallelizable = IS_PARALLELIZABLE
llm_policy.model_parallel = MODEL_PARALLEL
llm_policy.config.use_cache = USE_CACHE
llm_policy_config = LoraConfig(target_modules=TARGET_MODULES_policy,
                               lora_dropout=LORA_DROPOUT,
                               lora_alpha=LORA_ALPHA,
                               task_type='CAUSAL_LM',
                               bias="none",
                               r=LORA_R)
llm_model_policy = get_peft_model(llm_policy, llm_policy_config).to(device)
mlp_policy = MLP_P().to(device)
policy_model = MergeModel_P(llm_model_policy, mlp_policy, sequence_length_policy).to(device)
llm_value = LLMModel.from_pretrained(PATH_MODEL_PRETRAIN_value, torch_dtype=torch.bfloat16)
llm_value.is_parallelizable = IS_PARALLELIZABLE
llm_value.model_parallel = MODEL_PARALLEL
llm_value.config.use_cache = USE_CACHE
llm_value_config = LoraConfig(target_modules=TARGET_MODULES_value,
                               lora_dropout=LORA_DROPOUT,
                               lora_alpha=LORA_ALPHA,
                               task_type='CAUSAL_LM',
                               bias="none",
                               r=LORA_R)
llm_model_value = get_peft_model(llm_value, llm_policy_config).to(device)
mlp_value = MLP_V().to(device)
value_model = MergeModel_V(llm_model_value, mlp_value, sequence_length_value).to(device)

student_model = H2GCN(feat_dim=features.size(1), hidden_dim=128, class_dim=5).to(device)
student_model.load_state_dict(torch.load('{}_student_best_model_{}.pt'.format(dataset_name, label_rate)))
student_model = student_model.to(device)

optimizer_s = optim.Adam(student_model.parameters(), lr=0.0001, weight_decay=1e-6)
optimizer_policy = optim.Adam(policy_model.parameters(), lr=0.0001, weight_decay=1e-6)
optimizer_value = optim.Adam(value_model.parameters(), lr=0.0001, weight_decay=1e-6)

# scaler = GradScaler()

agent = PPOAgent(policy_model, value_model, optimizer_policy, optimizer_value)

for epoch in range(300):
    trajectories = []
    prompts, logits = prompt_collections(dataset_name, label_rate)

    logits = logits.to(device)
    # for i, prompt in enumerate(prompts):
    for index in indices:
        prompt = prompts[index]
        inputs_policy = tokenizer_policy(prompt,
                                         return_tensors="pt",  # 返回 PyTorch tensor 形式，或者选择 'tf' 返回 TensorFlow tensor
                                         padding="max_length",  # 对齐文本长度以适应模型输入的固定长度
                                         truncation=True,  # 截断长于 max_length 的文本
                                         max_length=128  # 设定最大输入长度，例如 64 个 token
                                         ).to(device)
        inputs_value = tokenizer_value(prompt,
                                         return_tensors="pt",  # 返回 PyTorch tensor 形式，或者选择 'tf' 返回 TensorFlow tensor
                                         padding="max_length",  # 对齐文本长度以适应模型输入的固定长度
                                         truncation=True,  # 截断长于 max_length 的文本
                                         max_length=128  # 设定最大输入长度，例如 64 个 token
                                         ).to(device)
        # with autocast():
        value = agent.value_net(**inputs_value)
        action, log_prob = agent.take_action(inputs_policy)
        assignment = F.one_hot(torch.tensor(action), num_classes=4).float().to(device)
        reward = distill(index, logits, student_model, features, edge_index, assignment, labels, optimizer_s, epoch, mask)

        trajectories.append((inputs_policy, inputs_value, action, log_prob, reward, value, epoch))

    agent.update(trajectories)

    with open('{}_pseudo_part_{}.txt'.format(dataset_name, label_rate), 'a') as f:

        with torch.no_grad():
            student_logits, preds, _ = student_model(eidx_to_sp(len(features), edge_index.detach().cpu()).to(device),
                                                     features)
            acc = compute_accuracy_teacher(preds, labels_true)
            f.write(str(acc))
            f.write('\n')
            f.close()


























