import numpy as np
import torch
import random
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from sklearn.preprocessing import normalize
import json
import pandas as pd

pubmed_mapping = {
    0: "Experimentally induced diabetes",
    1: "Type 1 diabetes",
    2: "Type 2 diabetes"
}

def get_pubmed_casestudy(corrected=False, SEED=0):
    _, data_X, data_Y, data_pubid, data_edges = parse_pubmed()
    data_X = normalize(data_X, norm='l1')

    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    data_name = 'PubMed'

    dataset = Planetoid('dataset/dataset', data_name, transform=T.NormalizeFeatures())
    data = dataset[0]

    data.x = torch.tensor(data_X)
    data.edge_index = torch.tensor(data_edges)
    data.y = torch.tensor(data_Y)

    # split data
    node_id = np.arange(data.num_nodes)
    np.random.shuffle(node_id)

    num_nodes = data.num_nodes
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    num_classes = 3

    train_idx = []
    val_idx = []
    test_idx = []
    labels = data['y']

    for i in range(num_classes):
        class_idx = torch.where(labels == i)[0]
        assert class_idx.size(0) >= 20, f"Not enough nodes for class {i}"

        permuted_idx = class_idx[torch.randperm(class_idx.size(0))]

        train_idx.extend(permuted_idx[: 20].tolist())

    remaining_idx = list(set(range(data.num_nodes)) - set(train_idx))
    np.random.shuffle(remaining_idx)

    val_idx = remaining_idx[:500]
    test_idx = remaining_idx[500: 1500]

    assert len(val_idx) == 500, "Not enough nodes for validation set"
    assert len(test_idx) == 1000, "Not enough nodes for test set"

    train_idx = torch.tensor(train_idx, dtype=torch.long)
    val_idx = torch.tensor(val_idx, dtype=torch.long)
    test_idx = torch.tensor(test_idx, dtype=torch.long)

    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    return data, data_pubid


def parse_pubmed():
    path = './dataset/dataset/PubMed/data/'

    n_nodes = 19717
    n_features = 500

    data_X = np.zeros((n_nodes, n_features), dtype='float32')
    data_Y = [None] * n_nodes
    data_pubid = [None] * n_nodes
    data_edges = []

    paper_to_index = {}
    feature_to_index = {}

    with open(path+'Pubmed-Diabetes.NODE.paper.tab', 'r') as node_file:
        node_file.readline()
        node_file.readline()

        k = 0

        for i, line in enumerate(node_file.readlines()):
            items = line.strip().split('\t')

            paper_id = items[0]
            data_pubid[i] = paper_id
            paper_to_index[paper_id] = i

            label = int(items[1].split('=')[-1]) - 1
            data_Y[i] = label

            features = items[2: -1]
            for feature in features:
                parts = feature.split('=')
                fname = parts[0]
                fvalue = float(parts[1])

                if fname not in feature_to_index:
                    feature_to_index[fname] = k
                    k += 1

                data_X[i, feature_to_index[fname]] = fvalue

    # parse graph
    data_A = np.zeros((n_nodes, n_nodes), dtype='float32')

    with open(path+'Pubmed-Diabetes.DIRECTED.cites.tab', 'r') as edge_file:

        edge_file.readline()
        edge_file.readline()

        for i, line in enumerate(edge_file.readlines()):
            items = line.strip().split('\t')
            edge_id = items[0]

            tail = items[1].split(':')[-1]
            head = items[3].split(':')[-1]

            data_A[paper_to_index[tail], paper_to_index[head]] = 1.0
            data_A[paper_to_index[head], paper_to_index[tail]] = 1.0
            if head != tail:
                data_edges.append((paper_to_index[head], paper_to_index[tail]))
                data_edges.append((paper_to_index[tail], paper_to_index[head]))
    return data_A, data_X, data_Y, data_pubid, np.unique(data_edges, axis=0).transpose()

def get_raw_text_pubmed(use_text=False, seed=0):
    data, data_pubid = get_pubmed_casestudy(SEED=seed)
    if not use_text:
        return data, None

    f = open('dataset/dataset/PubMed/pubmed.json')
    pubmed = json.load(f)
    df_pubmed = pd.DataFrame.from_dict(pubmed)

    AB = df_pubmed['AB'].fillna("")
    TI = df_pubmed['TI'].fillna("")
    text = {'title': [], 'abs': [], 'label': []}
    for ti, ab in zip(TI, AB):
        text['title'].append(ti)
        text['abs'].append(ab)

    for i in range(len(data.y)):
        text['label'].append(pubmed_mapping[data.y[i].item()])

    return data, text





















