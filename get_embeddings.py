# # from transformers import AutoTokenizer, AutoModel
# # import torch
# # from get_prompts import prompt_collections
# # import numpy as np
# # from config import LEARNING_RATE, EPOCHS, SAVE_STEPS, VAL_SET_SIZE, TARGET_MODULES_policy, TARGET_MODULES_value
# # from config import MICRO_BATCH_SIZE, BATCH_SIZE, GRADIENT_ACCUMULATION_STEPS
# # from config import PATH_MODEL_PRETRAIN_policy, PATH_MODEL_PRETRAIN_value, DATA_PATH, MODEL_SAVE_DIR, REPO_ID_policy, REPO_ID_value
# # from config import IS_PARALLELIZABLE, MODEL_PARALLEL, USE_CACHE
# # from config import MAX_LENGTH_Q, MAX_LENGTH_A, MAX_LENGTH_QA
# # from config import LORA_DROPOUT, LORA_ALPHA, LORA_R
# # from config import USE_CUDA, WEIGHT_DECAY
# # from config import USE_ALL_LOSS
# # import torch.nn.functional as F
# # from peft import LoraConfig, get_peft_model
# # from model_merge import MLP_P, MLP_V, MergeModel_P, MergeModel_V
# # from transformers import AutoTokenizer as LLMTokenizer
# # from transformers import LlamaForCausalLM as LLMModel
# # # 加载模型和分词器
# #
# # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # tokenizer_policy = LLMTokenizer.from_pretrained(PATH_MODEL_PRETRAIN_policy, add_eos_token=True, trust_remote_code=True)
# # ID_START = 128006
# # ID_END = 128007
# # ID_BOS = 128000
# # ID_EOS = 128009
# # ID_PAD = ID_EOS
# # ID_BR = 198  # "\n"
# # ID_SYSTEM = 9125
# # ID_MODEL = 78191
# # ID_USER = 882
# # tokenizer_policy.pad_token_id = ID_EOS
# # tokenizer_policy.eos_token_id = ID_EOS
# # tokenizer_policy.padding_side = "right"  # NO use attention-mask
# # out_dim_policy = 128256
# # out_dim_value = 128256
# # sequence_length_policy = 128
# # sequence_length_value = 128
# # llm_policy = LLMModel.from_pretrained(PATH_MODEL_PRETRAIN_policy, torch_dtype=torch.bfloat16).to(device)
# # llm_policy.is_parallelizable = IS_PARALLELIZABLE
# # llm_policy.model_parallel = MODEL_PARALLEL
# # llm_policy.config.use_cache = USE_CACHE
# #
# # dataset_name = 'pubmed'
# # label_rate = 1
# # prompts, logits = prompt_collections(dataset_name, label_rate)
# #
# # # batch_size = 16  # 根据GPU内存调整
# # all_embeddings = []
# # # 设置最大输入长度
# #
# # max_length = 2048
# # # 分批处理
# # # 其他代码保持不变
# #
# # # 分批处理
# # for i, prompt in enumerate(prompts):
# #     print(i)
# #     inputs_policy = tokenizer_policy(prompt,
# #                                      return_tensors="pt",
# #                                      padding="max_length",
# #                                      truncation=True,
# #                                      max_length=max_length
# #                                      ).to(device)
# #     with torch.no_grad():
# #         outputs = llm_policy(**inputs_policy, output_hidden_states=True)
# #     embeddings = outputs.hidden_states[-1][:, 0, :]  # 获取最后一层的 [CLS] tokens 的表示
# #     all_embeddings.append(embeddings)
# #
# # # 其他代码保持不变
# #
# # # 合并所有表示向量
# # # 合并所有表示向量
# # all_embeddings = torch.cat(all_embeddings, dim=0).detach().cpu()
# # all_embeddings = all_embeddings.float()  # 将数据类型转换为 float32
# # all_embeddings = all_embeddings.numpy()  # 转换为 NumPy 数组
# # np.save('{}_embeddings_{}.npy'.format(dataset_name, label_rate), all_embeddings)
#
#
# ID_START = 128006
# ID_END = 128007
# ID_BOS = 128000
# ID_EOS = 128009
# ID_PAD = ID_EOS
# ID_BR = 198  # "\n"
# ID_SYSTEM = 9125
# ID_MODEL = 78191
# ID_USER = 882
# from transformers import AutoTokenizer, AutoModel
# import torch
# from get_prompts import prompt_collections
# import numpy as np
# from config import PATH_MODEL_PRETRAIN_policy, DATA_PATH, REPO_ID_policy
# # from config import ID_START, ID_END, ID_BOS, ID_EOS, ID_PAD, ID_BR, ID_SYSTEM, ID_MODEL, ID_USER
# # from config import USE_CUDA, MAX_LENGTH
# from transformers import AutoTokenizer as LLMTokenizer
# from transformers import LlamaForCausalLM as LLMModel
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# tokenizer_policy = LLMTokenizer.from_pretrained(PATH_MODEL_PRETRAIN_policy, add_eos_token=True, trust_remote_code=True)
# tokenizer_policy.pad_token_id = ID_EOS
# tokenizer_policy.eos_token_id = ID_EOS
# tokenizer_policy.padding_side = "right"
#
# llm_policy = LLMModel.from_pretrained(PATH_MODEL_PRETRAIN_policy, torch_dtype=torch.bfloat16).to(device)
#
# dataset_name = 'pubmed'
# label_rate = 3
# prompts, logits = prompt_collections(dataset_name, label_rate)
#
# max_length = 2048
# all_embeddings = []
#
# batch_size = 8  # Adjust based on GPU memory
# num_batches = (len(prompts) + batch_size - 1) // batch_size
#
# for i in range(num_batches):
#     start_idx = i * batch_size
#     end_idx = (i + 1) * batch_size
#     batch_prompts = prompts[start_idx:end_idx]
#
#     inputs_policy = tokenizer_policy(batch_prompts,
#                                      return_tensors="pt",
#                                      padding="longest",
#                                      truncation=True,
#                                      max_length=max_length)
#     inputs_policy = {k: v.to(device) for k, v in inputs_policy.items()}
#
#     with torch.no_grad():
#         outputs = llm_policy(**inputs_policy, output_hidden_states=True)
#     embeddings = outputs.hidden_states[-1][:, 0, :].detach().cpu()
#     all_embeddings.append(embeddings)
#
# all_embeddings = torch.cat(all_embeddings, dim=0).float().numpy()
# np.save('{}_embeddings_{}.npy'.format(dataset_name, label_rate), all_embeddings)
#

# from transformers import AutoTokenizer, AutoModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from get_prompts import prompt_collections
import numpy as np
from config import PATH_MODEL_PRETRAIN_policy, DATA_PATH, REPO_ID_policy
# from config import PATH_MODEL_PRETRAIN_policy, DATA_PATH, REPO_ID_policy
# from config import ID_START, ID_END, ID_BOS, ID_EOS, ID_PAD, ID_BR, ID_SYSTEM, ID_MODEL, ID_USER
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ID_START = 128006
ID_END = 128007
ID_BOS = 128000
ID_EOS = 128009
ID_PAD = ID_EOS
ID_BR = 198  # "\n"
ID_SYSTEM = 9125
ID_MODEL = 78191
ID_USER = 882
tokenizer_policy = AutoTokenizer.from_pretrained(PATH_MODEL_PRETRAIN_policy, add_eos_token=True, trust_remote_code=True)
tokenizer_policy.pad_token_id = ID_EOS
tokenizer_policy.eos_token_id = ID_EOS
tokenizer_policy.padding_side = "right"

# 加载模型时使用 device_map="auto" 自动分配到 GPU
llm_policy = AutoModelForCausalLM.from_pretrained(
    PATH_MODEL_PRETRAIN_policy,
    torch_dtype=torch.bfloat16,
    device_map="auto"  # 自动分配到 GPU
)

dataset_name = 'amazon'
label_rate = 3
prompts, logits = prompt_collections(dataset_name, label_rate)

max_length = 1024
all_logits = []

batch_size = 4  # Adjust based on GPU memory
num_batches = (len(prompts) + batch_size - 1) // batch_size

for i in range(num_batches):
    start_idx = i * batch_size
    end_idx = (i + 1) * batch_size
    batch_prompts = prompts[start_idx:end_idx]

    inputs_policy = tokenizer_policy(batch_prompts,
                                     # prompt,
                                     return_tensors="pt",  # 返回 PyTorch tensor 形式，或者选择 'tf' 返回 TensorFlow tensor
                                     padding="max_length",  # 对齐文本长度以适应模型输入的固定长度
                                     truncation=True,  # 截断长于 max_length 的文本
                                     max_length=max_length)
    inputs_policy = {k: v.to(device) for k, v in inputs_policy.items()}
    print("[{}/{}]".format(i, num_batches))
    with torch.no_grad():
        outputs = llm_policy(**inputs_policy)
    # logits_batch = outputs.logits.detach().cpu()  # 获取最后一个 token 的 logits
    logits_batch = outputs.logits.detach().cpu().float().numpy()  # 获取最后一个 token 的 logits
    np.save('embeddings/{}_embeddings_{}_{}.npy'.format(dataset_name, label_rate, i), logits_batch)

#     all_logits.append(logits_batch)
#
# all_logits = torch.cat(all_logits, dim=0).float().numpy()
# np.save('embeddings/{}_embeddings_{}.npy'.format(dataset_name, label_rate), all_logits)





