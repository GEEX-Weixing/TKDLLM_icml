import numpy as np
import networkx as nx

def get_text_similar_matrix(embeddings):
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized_embeddings = embeddings / norms

    # Compute cosine similarity using dot product
    similarity_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)

    # Set the diagonal elements to 0
    np.fill_diagonal(similarity_matrix, 0)

    return similarity_matrix


def get_indices_list(logit_1, logit_2, logit_3, logit_4):
# print(logit_4)
    similarity_matrix_1 = get_text_similar_matrix(logit_1)
    similarity_matrix_2 = get_text_similar_matrix(logit_2)
    similarity_matrix_3 = get_text_similar_matrix(logit_3)
    similarity_matrix_4 = get_text_similar_matrix(logit_4)

    indices_list = []

    for i in range(logit_1.shape[0]):
        nei_indices = []
        sorted_similarity_indices_1 = np.arange(logit_1.shape[0])[np.argsort(-similarity_matrix_1[i])]
        sorted_similarity_indices_2 = np.arange(logit_1.shape[0])[np.argsort(-similarity_matrix_2[i])]
        sorted_similarity_indices_3 = np.arange(logit_1.shape[0])[np.argsort(-similarity_matrix_3[i])]
        sorted_similarity_indices_4 = np.arange(logit_1.shape[0])[np.argsort(-similarity_matrix_4[i])]

        top_4_indices_1 = sorted_similarity_indices_1[:2].tolist()
        top_4_indices_2 = sorted_similarity_indices_2[:2].tolist()
        top_4_indices_3 = sorted_similarity_indices_3[:2].tolist()
        top_4_indices_4 = sorted_similarity_indices_4[:2].tolist()

        nei_indices.append(top_4_indices_1)
        nei_indices.append(top_4_indices_2)
        nei_indices.append(top_4_indices_3)
        nei_indices.append(top_4_indices_4)

        nei_indices = list(set(num for row in nei_indices for num in row))
        indices_list.append(nei_indices)
    return indices_list


def compute_ppr_matrix(G, alpha=0.85):
    num_nodes = G.number_of_nodes()

    # 初始化一个 N x N 的矩阵
    ppr_matrix = np.zeros((num_nodes, num_nodes))
    # 遍历每个节点，计算相对于该节点的 Personalized PageRank
    for node in G.nodes():
        # 设置节点本身作为 personalized vector (teleporting vector)
        personalized = {node: 1}

        # 计算 PPR 分数
        ppr = nx.pagerank(G, alpha=alpha, personalization=personalized)

        # 将 PPR 分数存入矩阵中
        for target_node, score in ppr.items():
            ppr_matrix[node, target_node] = score

    return ppr_matrix


def get_ppr_neighbor_indices(ppr_matrix):

    np.fill_diagonal(ppr_matrix, 0)
    neighbors_list = []
    for i in range(ppr_matrix.shape[0]):
        # sorted_neighbor = np.arange(ppr_matrix.shape[0])[np.argsort(-ppr_matrix[i])]
        # top_neighbors = sorted_neighbor[:12].tolist()
        top_neighbors = np.argsort(ppr_matrix[i])[::-1][:12]
        neighbors_list.append(top_neighbors)
    return neighbors_list




