from neighbor_select import get_indices_list
import torch
import numpy as np
from get_paper_txt import get_text

def prompt_collections(dataset, label_rate):
    dataset_name = dataset
    if dataset == 'amazon':
        logit_1 = np.load('logit/{}/dirgnn_{}.npy'.format(label_rate, dataset))
        logit_2 = np.load('logit/{}/h2gcn_{}.npy'.format(label_rate, dataset))
        logit_3 = np.load('logit/{}/holognn_{}.npy'.format(label_rate, dataset))
        logit_4 = np.load('logit/{}/gprgnn_{}.npy'.format(label_rate, dataset))
    elif dataset == 'cora':
        logit_1 = np.load('outputs/{}/{}/appnp_{}.npy'.format(dataset_name, label_rate, dataset_name))
        logit_2 = np.load('outputs/{}/{}/gcn_{}.npy'.format(dataset_name, label_rate, dataset_name))
        logit_3 = np.load('outputs/{}/{}/h2gcn_{}.npy'.format(dataset_name, label_rate, dataset_name))
        logit_4 = np.load('outputs/{}/{}/gcn2_{}.npy'.format(dataset_name, label_rate, dataset_name))
    elif dataset == 'pubmed':
        logit_1 = np.load('outputs/{}/{}/appnp_{}.npy'.format(dataset_name, label_rate, dataset_name))
        logit_2 = np.load('outputs/{}/{}/gcn_{}.npy'.format(dataset_name, label_rate, dataset_name))
        logit_3 = np.load('outputs/{}/{}/gat_{}.npy'.format(dataset_name, label_rate, dataset_name))
        logit_4 = np.load('outputs/{}/{}/h2gcn_{}.npy'.format(dataset_name, label_rate, dataset_name))
    elif dataset == 'wiki_cs':
        logit_1 = np.load('outputs/{}/{}/appnp_{}.npy'.format(dataset_name, label_rate, dataset_name))
        logit_2 = np.load('outputs/{}/{}/gcn_{}.npy'.format(dataset_name, label_rate, dataset_name))
        logit_3 = np.load('outputs/{}/{}/gat_{}.npy'.format(dataset_name, label_rate, dataset_name))
        logit_4 = np.load('outputs/{}/{}/gcn2_{}.npy'.format(dataset_name, label_rate, dataset_name))

    neighbors_list = get_indices_list(logit_1, logit_2, logit_3, logit_4)
    logits = torch.tensor(np.stack([logit_1, logit_2, logit_3, logit_4], axis=0))
    # with open('amazon_5000_des.txt', 'r') as g:
    #     amazon_short_texts = g.read().split('\n')
    if dataset_name == 'wiki_cs':
        with open('wikics_tokens.txt') as f:
            clean_texts = f.read().split('\n\n')
    elif dataset_name == 'amazon':
        with open('amazon_5000_des.txt', 'r') as g:
            clean_texts = g.read().split('\n')
    else:
        data, clean_texts = get_text(dataset_name)

    messages = []

    for i in range(len(clean_texts)):
        node_index = i
        text = clean_texts[node_index]
        if dataset == 'cornell' or 'texas' or 'wisconsin' or 'washington':
            role_2 = {"Semantic attributes": "It is the content description of this target webpage: {" + text + "}.",
                      "Prediction attributes": "The dirgnn's logits output of this target webpage is {" + str(logit_1[node_index]) + "}."
                                 "The gprgnn's logits output of this target webpage is {" + str(logit_4[node_index]) + "}. "
                                 "The h2gcn's logits output of this target webpage is {" + str(logit_2[node_index]) + "}."
                                 "The holognn's logits output of this target webpage is {" + str(logit_3[node_index]) + "}. ",
                      "Structural attributes": "It has following important neighbors (webpages), which are closely related the target webpage. Their content descriptions are: " + "."}

        elif dataset == 'amazon':
            role_2 = {"Semantic attributes": "It is the content description of this target product: {" + text + "}.",
                      "Prediction attributes": "The dirgnn's logits output of this target product is {" + str(
                          logit_1[node_index]) + "}."
                                                 "The h2gcn's logits output of this target product is {" + str(
                          logit_4[node_index]) + "}. "
                                                 "The holonet's logits output of this target product is {" + str(
                          logit_2[node_index]) + "}."
                                                 "The gprgnn's logits output of this target product is {" + str(
                          logit_3[node_index]) + "}. ",
                      "Structural attributes": "It has following important neighbors (products), which are closely related the target product. Their content descriptions are: " + "."}
        elif dataset == 'cora':
            role_2 = {"Semantic attributes": "It is the content description of this target paper: {" + text + "}.",
                      "Prediction attributes": "The appnp's logits output of this target paper is {" + str(
                          logit_1[node_index]) + "}."
                                                 "The gcn's logits output of this target paper is {" + str(
                          logit_4[node_index]) + "}. "
                                                 "The h2gcn's logits output of this target paper is {" + str(
                          logit_2[node_index]) + "}."
                                                 "The gcn2's logits output of this target paper is {" + str(
                          logit_3[node_index]) + "}. ",
                      "Structural attributes": "It has following important neighbors (papers), which are closely related the target paper. Their content descriptions are: " + "."}
        elif dataset == 'pubmed':
            role_2 = {"Semantic attributes": "It is the content description of this target paper: {" + text + "}.",
                      "Prediction attributes": "The appnp's logits output of this target paper is {" + str(
                          logit_1[node_index]) + "}."
                                                 "The gcn's logits output of this target paper is {" + str(
                          logit_4[node_index]) + "}. "
                                                 "The gat's logits output of this target paper is {" + str(
                          logit_2[node_index]) + "}."
                                                 "The h2gcn's logits output of this target paper is {" + str(
                          logit_3[node_index]) + "}. ",
                      "Structural attributes": "It has following important neighbors (papers), which are closely related the target paper. Their content descriptions are: " + "."}
        elif dataset == 'wiki_cs':
            role_2 = {"Semantic attributes": "It is the content description of this target wabpage: {" + text + "}.",
                      "Prediction attributes": "The appnp's logits output of this target wabpage is {" + str(
                          logit_1[node_index]) + "}."
                                                 "The gcn's logits output of this target wabpage is {" + str(
                          logit_4[node_index]) + "}. "
                                                 "The gat's logits output of this target wabpage is {" + str(
                          logit_2[node_index]) + "}."
                                                 "The gcn2's logits output of this target wabpage is {" + str(
                          logit_3[node_index]) + "}. ",
                      "Structural attributes": "It has following important neighbors (wabpages), which are closely related the target wabpage. Their content descriptions are: " + "."}
        neighbor_list = neighbors_list[node_index]
        for j in range(len(neighbor_list)):
            neighbor_index = neighbor_list[j]
            neighbor_text = clean_texts[neighbor_index]
            role_2['Structural attributes'] += "{[Index: " + str(neighbor_index) + "]. Its description is [" + neighbor_text + "]}."
        messages.append(role_2)
    return messages, logits

def get_total_prompts(dataset, label_rate, tokenizer):
    messages, logits = prompt_collections(dataset, label_rate)
    dataset_name = dataset
    if dataset_name == 'amazon':
        system_prompt = [{"role": "system", "content": """
                                There are four names of teacher networks: ['appnp', 'gcn', 'gat', 'h2gcn']. We need to perform knowledge distillation for each node in this graph consist of nodes (products) and edges (co-purchase relationships). 
                        You will serve as an assistant to help me to assign the best teacher network for the target node (product) based on the following information.
                        I will provide you with three kinds of attributes of the target node (product). Here are the instructions:
                        1. I will provide you with information in the form of a JSON string that describes the node (product):
                        {"Semantic attributes": "The description of the product's purchase information", 
                        "Prediction attributes": "Four teacher networks' logit output of this target node", 
                        "Structural attributes": "Important neighbors (products), which are closely related the target node (product) and their purchase information."}

                        Requirements:
                        1. Please provide your response in JSON format, following this structure:
                           {
                             "reasoning": "Briefly explain your reasoning process for the predicted category",
                             "teacher network": "The best teacher network you assign for this node (product), this result must belong to these 4 teachers: ['appnp', 'gcn', 'gat', 'h2gcn']."
                           }
                        2. There are 2000 words limits for the "reasoning".
                        3. Do not provide any other text outside the JSON string.
                        4. Focus only on content in the actual text and avoid making false associations.
                        5. The output can only contain "teacher network" and "reasoning".
                    """}]
    elif dataset_name == 'wiki_cs':
        system_prompt = [{"role": "system", "content": """
                                There are four names of teacher networks: ['appnp', 'gcn', 'gat', 'gcn2']. We need to perform knowledge distillation for each node in this graph consist of nodes (websages) and edges (link relationships). 
                        You will serve as an assistant to help me to assign the best teacher network for the target node (websage) based on the following information.
                        I will provide you with three kinds of attributes of the target node (websage). Here are the instructions:
                        1. I will provide you with information in the form of a JSON string that describes the node (websage):
                        {"Semantic attributes": "The description of the websage", 
                        "Prediction attributes": "Four teacher networks' logit output of this target node", 
                        "Structural attributes": "Important neighbors (websages), which are closely related the target node (websage) and their descriptions."}

                        Requirements:
                        1. Please provide your response in JSON format, following this structure:
                           {
                             "reasoning": "Briefly explain your reasoning process for the predicted category",
                             "teacher network": "The best teacher network you assign for this node (product), this result must belong to these 4 teachers: ['appnp', 'gcn', 'gat', 'gcn2']."
                           }
                        2. There are 2000 words limits for the "reasoning".
                        3. Do not provide any other text outside the JSON string.
                        4. Focus only on content in the actual text and avoid making false associations.
                        5. The output can only contain "teacher network" and "reasoning".
                    """}]
    elif dataset_name == 'cora':
        system_prompt = [{"role": "system", "content": """
                                There are four names of teacher networks: ['appnp', 'gcn', 'h2gcn', 'gcn2']. We need to perform knowledge distillation for each node in this graph consist of nodes (papers) and edges (citation relationships). 
                        You will serve as an assistant to help me to assign the best teacher network for the target node (paper) based on the following information.
                        I will provide you with three kinds of attributes of the target node (paper). Here are the instructions:
                        1. I will provide you with information in the form of a JSON string that describes the node (paper):
                        {"Semantic attributes": "The title and abstract of this paper", 
                        "Prediction attributes": "Four teacher networks' logit output of this target node", 
                        "Structural attributes": "Important neighbors (papers), which are closely related the target node (paper) and their contents."}

                        Requirements:
                        1. Please provide your response in JSON format, following this structure:
                           {
                             "reasoning": "Briefly explain your reasoning process for the predicted category",
                             "teacher network": "The best teacher network you assign for this node (paper), this result must belong to these 4 teachers: ['appnp', 'gcn', 'h2gcn', 'gcn2']."
                           }
                        2. There are 2000 words limits for the "reasoning".
                        3. Do not provide any other text outside the JSON string.
                        4. Focus only on content in the actual text and avoid making false associations.
                        5. The output can only contain "teacher network" and "reasoning".
                    """}]
    elif dataset_name == 'pubmed':
        system_prompt = [{"role": "system", "content": """
                                There are four names of teacher networks: ['appnp', 'gcn', 'gat', 'gcn2']. We need to perform knowledge distillation for each node in this graph consist of nodes (papers) and edges (citation relationships). 
                        You will serve as an assistant to help me to assign the best teacher network for the target node (paper) based on the following information.
                        I will provide you with three kinds of attributes of the target node (paper). Here are the instructions:
                        1. I will provide you with information in the form of a JSON string that describes the node (paper):
                        {"Semantic attributes": "The title and abstract of this paper", 
                        "Prediction attributes": "Four teacher networks' logit output of this target node", 
                        "Structural attributes": "Important neighbors (papers), which are closely related the target node (paper) and their contents."}

                        Requirements:
                        1. Please provide your response in JSON format, following this structure:
                           {
                             "reasoning": "Briefly explain your reasoning process for the predicted category",
                             "teacher network": "The best teacher network you assign for this node (paper), this result must belong to these 4 teachers: ['appnp', 'gcn', 'h2gcn', 'gcn2']."
                           }
                        2. There are 2000 words limits for the "reasoning".
                        3. Do not provide any other text outside the JSON string.
                        4. Focus only on content in the actual text and avoid making false associations.
                        5. The output can only contain "teacher network" and "reasoning".
                    """}]

    inputs = [tokenizer.apply_chat_template(system_prompt + [{"role": "user", "content": messages[i]}], tokenize=False,
                                            add_generation_prompt=True) for i in range(len(messages))]
    return inputs, logits

