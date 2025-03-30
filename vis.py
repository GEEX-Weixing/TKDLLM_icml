import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from models.h2gcn import H2GCN
from sklearn.metrics.pairwise import cosine_similarity
from models.gcn import GCN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset_name = 'wisconsin'
label_rate = 3
features = torch.tensor(np.load('{}_feature.npy'.format(dataset_name))).cuda()
edge_index = torch.tensor(np.load('{}_edge_index.npy'.format(dataset_name))).cuda()
labels_true = torch.tensor(np.load('{}_labels.npy'.format(dataset_name))).cuda()
# student_model = GCN(features.size(1), hidden_dim=128, out_dim=len(labels_true.unique())).to(device)
student_model = H2GCN(feat_dim=features.size(1), hidden_dim=128, class_dim=5).to(device)
student_model.load_state_dict(torch.load('{}_student_best_model_{}.pt'.format(dataset_name, label_rate), map_location=device))
student_model.eval()


# 获取节点的logits输出
with torch.no_grad():
    logits, _, _ = student_model(features, edge_index)
# logits = logits
# 按照类别对节点进行排序
sorted_indices = torch.argsort(labels_true)
sorted_logits = logits[sorted_indices]

# 计算余弦相似度矩阵
similarity_matrix = cosine_similarity(sorted_logits.detach().cpu().numpy())

# 自定义调色板
custom_cmap = sns.diverging_palette(240, 10, as_cmap=True)  # 从深蓝色到浅黄色

# 可视化相似度矩阵
plt.figure(figsize=(10, 8))
sns.heatmap(similarity_matrix, cmap=custom_cmap, xticklabels=False, yticklabels=False)
# plt.title('Node Logit Similarity Matrix (Sorted by Class)')
plt.savefig('{}_{}_vis.png'.format(dataset_name, label_rate))
plt.show()