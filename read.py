import numpy as np
import torch
import random

from torch_geometric.data import Data
import os
from pathlib import Path
import re
from torch_geometric.utils import remove_self_loops

def parse_webkb(dataset):
    path = '/share/home/tj20526/data/xing/LLaMA3-SFT-master/llama3_sft/ft_llama3/web/{}'.format(dataset)
    # print(path)
    webpage_features_labels = np.genfromtxt("{}.content".format(path), dtype=np.dtype(str))
    data_X = webpage_features_labels[:, 1: -1].astype(np.float32)
    labels = webpage_features_labels[:, -1]

    class_map = {x: i for i, x in enumerate(['course', 'faculty', 'student','project', 'staff'])}
    data_Y = np.array([class_map[x] for x in labels])

    data_webpage_url = webpage_features_labels[:, 0]
    data_webpage_id_map = {x: i for i, x in enumerate(data_webpage_url)}
    edges_unordered = np.genfromtxt("{}.cites".format(path), dtype=np.dtype(str))

    edges = np.array(list(map(data_webpage_id_map.get, edges_unordered.flatten())), dtype=np.int32)

    # data_edges = np.array(edges[~(edges == None).max(0)], dtype=np.int32)
    # data_edges = np.vstack((data_edges, np.fliplr(data_edges)))
    odd_index_elements = edges[::2]  # [1, 3, 5, 7]

    # 提取偶数索引的元素 (即从1开始的偶数位置)
    even_index_elements = edges[1::2]  # [2, 4, 6, 8]

    # 将它们组合为2*4的数组
    data_edges = np.vstack((odd_index_elements, even_index_elements))

    return data_X, data_Y, data_webpage_url, torch.tensor(data_edges, dtype=torch.long).contiguous()

def get_webkb_casestudy(dataset, SEED=0):
    data_X, data_Y, data_webpage_url, data_edges = parse_webkb(dataset)

    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)  # Numpy module.
    random.seed(SEED)  # Python random module.

    data = Data(x=torch.tensor(data_X).float(), edge_index=torch.tensor(data_edges).long(),
                y=torch.tensor(data_Y).long(), num_nodes=len(data_Y))

    return data, data_webpage_url

def html_process(input_string):
    lines = input_string.split('\n')
    clean_text = ' '.join(lines[6:])

    tag_list = ['<.*?>', '\n', r'<a\s+href\s*=\s*".*?"\s*>', r'<IMG\s+SRC\s*=\s*".*?"\s+ALT\s*=\s*".*?"\s*>']
    for tag in tag_list:
        clean_text = re.sub(tag, '', clean_text, flags=re.IGNORECASE)
    clean_text = re.sub(r'\s+', '', clean_text)
    return clean_text

def delete_vacant_webpage(data, i):
    data.y = torch.cat((data.y[: i], data.y[(i+1):]))
    data.x = torch.cat((data.x[:i], data.x[(i+1):]))
    data.num_nodes -= 1
    mask = (data.edge_index[0] == i) | (data.edge_index[1] == i)
    data.edge_index = data.edge_index[:, ~mask]
    return data

def get_raw_text_webkb(dataset, use_text=False, seed=0):
    data, data_webpage_url = get_webkb_casestudy(dataset, seed)

    text = []
    clean_text = []
    category_list = ['course', 'faculty', 'student', 'project', 'staff']

    path = '/share/home/tj20526/data/xing/LLaMA3-SFT-master/llama3_sft/ft_llama3/page-text'
    pages_to_remove = []
    for i, url in enumerate(data_webpage_url):
        label = data.y[i]
        url = url.replace('/', '^')

        if not url.endswith('.html'):
            url += '^'
        file_path = '{}/{}/{}/{}'.format(path, category_list[label], dataset, url)
        if os.path.exists(file_path):
            t = open(file_path, 'r', errors='ignore').read()
            text.append(t)
        else:
            # print(i)
            pages_to_remove.append(i)

    # if dataset == 'wisconsin':
    #     pages_to_remove = [3, 5]
    # elif dataset == 'cornell':
    #     pages_to_remove = [12]
    # elif dataset == 'texas':
    #     pages_to_remove = [0]
    # elif dataset == 'washington':
    #     pages_to_remove = [1, 152, 156, 170, 171, 178, 214, 227]

    for i in reversed(pages_to_remove):
        data = delete_vacant_webpage(data, i)
    edge_index = data.edge_index
    out_of_range_edges = (edge_index[0] < 0) | (edge_index[0] >= data.num_nodes) | (edge_index[1] < 0) | (edge_index[1] >= data.num_nodes)

    data.edge_index = data.edge_index[:, ~out_of_range_edges]

    data.edge_index, _ = remove_self_loops(data.edge_index)

    # split data
    node_id = np.arange(data.num_nodes)
    np.random.shuffle(node_id)

    data.train_id = np.sort(node_id[:int(data.num_nodes * 0.2)])
    data.val_id = np.sort(
        node_id[int(data.num_nodes * 0.07):int(data.num_nodes * 0.2)])
    data.test_id = np.sort(node_id[int(data.num_nodes * 0.2):])

    data.train_mask = torch.tensor(
        [x in data.train_id for x in range(data.num_nodes)])
    data.val_mask = torch.tensor(
        [x in data.val_id for x in range(data.num_nodes)])
    data.test_mask = torch.tensor(
        [x in data.test_id for x in range(data.num_nodes)])
    if not use_text:
        return data, None
    for t in text:
        clean = html_process(t)
        clean_text.append(clean)
    data = Data(x=data.x, edge_index=data.edge_index, y=data.y, num_nodes=data.num_nodes,
                train_mask=data.train_mask, val_mask=data.val_mask, test_mask=data.test_mask)
    return data, clean_text


# data, clean_text = get_raw_text_webkb('texas', use_text=True, seed=0)
# # print(data)
# # print(len(clean_text))
# print(clean_text)












