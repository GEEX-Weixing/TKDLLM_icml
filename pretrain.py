import torch
from models.h2gcn import H2GCN
from models.gcn import GCN
from read import get_raw_text_webkb
from data_amazon import get_data
from get_prompts import prompt_collections
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import numpy as np

# from teacher_training_selection_ppo import compute_accuracy_teacher

def compute_accuracy_teacher(prediction, label):
    # _, prediction = student_model(feature, edge_index)
    correct = (prediction == label).sum().item()
    accuracy = correct / label.size(0) * 100
    return accuracy

def eidx_to_sp(n: int, edge_index: torch.Tensor, device=None) -> torch.sparse.Tensor:
    indices = edge_index
    values = torch.FloatTensor([1.0] * len(edge_index[0])).to(edge_index.device)
    coo = torch.sparse_coo_tensor(indices=indices, values=values, size=[n, n])
    if device is None:
        device = edge_index.device
    return coo.to(device)

# device = torch.device("")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset_name = 'amazon'
label_rate = 1
# data, _ = get_raw_text_webkb(dataset_name, use_text=True, seed=0)
data = get_data()
features = torch.tensor(np.load('{}_feature.npy'.format(dataset_name))).to(device)
edge_index = torch.tensor(np.load('{}_edge_index.npy'.format(dataset_name))).to(device)
labels = torch.tensor(np.load('{}_labels.npy'.format(dataset_name))).to(device)
model = H2GCN(feat_dim=features.size(1), hidden_dim=128, class_dim=5).to(device)
# model = GCN(features.size(1), 128, 7)
# model.load_state_dict(torch.load('cornell_student_best_model.pt'))
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

_, logits_t = prompt_collections(dataset_name, label_rate)
logits_t = logits_t.mean(dim=0)
# teacher_prob = torch.nn.functional.softmax(selected_logit, dim=-1)
logits_t = torch.nn.functional.softmax(logits_t, dim=-1).to(device)
best_acc = 0.

for epoch in range(400):
    logits, prediction, _ = model(eidx_to_sp(len(features), edge_index.detach().cpu()).to(device), features)
    # logits, prediction, _ = model(features, edge_index)
    ce_loss = F.nll_loss(logits[data.train_mask1[0]], labels[data.train_mask1[0]])
    logits = torch.nn.functional.softmax(logits, dim=-1)
    kl_loss = F.kl_div(logits, logits_t, reduction='batchmean')
    final_loss = 0.5 * ce_loss + 0.5 * kl_loss

    final_loss.backward()
    optimizer.step()

    acc = compute_accuracy_teacher(prediction, labels)

    print("Epoch: {}/{} CE Loss: {} KL Loss: {} Accuracy: {}".format(epoch, 400, ce_loss, kl_loss.item(), acc))

    if acc >= best_acc:
        best_acc = acc
        torch.save(model.state_dict(), '{}_student_best_model_{}_h2gcn.pt'.format(dataset_name, label_rate))

# with torch.no_grad():
#     logits, prediction, _ = model(eidx_to_sp(len(features), edge_index.detach().cpu()), features)
#     acc = compute_accuracy_teacher(prediction, labels)
#     print(acc)

