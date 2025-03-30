import torch
from vllm import LLM, SamplingParams
from neighbor_select import get_indices_list
from get_paper_txt import get_text
import numpy as np
# from get_prompts import prompt_collections
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from get_vllm_prompts import prompt_collections
import warnings
warnings.filterwarnings("ignore")

dataset_name = 'amazon'
label_rate = 5
message, logit = prompt_collections(dataset_name, label_rate)
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
model = '/ssdfs/datahome/tj20526/LLMs/meta-llama/Llama-3.1-8B-Instruct'
tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False, padding_side='left')
sampling_params = SamplingParams(temperature=0.7, top_p=0.8, repetition_penalty=1.05, max_tokens=512, min_tokens=128)
llm = LLM(model=model, trust_remote_code=False, dtype=torch.bfloat16)
inputs = [tokenizer.apply_chat_template(system_prompt + [{"role": "user", "content": message[i]}], tokenize=False,
                                        add_generation_prompt=True) for i in range(len(message))]
outputs = llm.generate(inputs, sampling_params=sampling_params)

with open('{}_{}_selection.txt'.format(dataset_name, label_rate), 'a') as d:
    for output in outputs:
        generated_text = output.outputs[0].text
        d.write(generated_text)
        d.write('\n\n\n\n')
    print('Finish!')
