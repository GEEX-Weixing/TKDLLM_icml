# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2023/3/5 21:29
# @author  : Mo
# @function: config of llama3


# optimized for RTX 4090. for larger GPUs, increase some of these?
MICRO_BATCH_SIZE = 4  # 4  # default=4  # this could actually be 5 but i like powers of 2
BATCH_SIZE = 1  # 128
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
LEARNING_RATE = 1e-5  ### 5e-5
# LEARNING_RATE = 3e-4  # default=3e-4  # the Karpathy constant
EPOCHS = 3  # default=3  # we don't always need 3 tbh
# LORA_DROPOUT = 0.1
# LORA_ALPHA = 32
# LORA_R = 32
WEIGHT_DECAY = 0.01
LORA_DROPOUT = 0.1
LORA_ALPHA = 2
LORA_R = 1
SAVE_STEPS = 128
VAL_SET_SIZE = 0
MAX_LENGTH_Q = 128 - 1  # default=128 - 2
MAX_LENGTH_A = 128 - 1  # default=128 - 2
MAX_LENGTH_QA = MAX_LENGTH_Q + MAX_LENGTH_A + 2
TARGET_MODULES_policy = ["q_proj",
                  "k_proj",
                  "v_proj",
                  "o_proj",  # q,k,v,o参数计算
                  "down_proj",  # FFN层计算
                  "gate_proj",
                  "up_proj",
                  ]
TARGET_MODULES_value = ["q_proj",
                  "k_proj",
                  "v_proj",
                  "o_proj",  # q,k,v,o参数计算
                  "down_proj",  # FFN层计算
                  "gate_proj",
                  "up_proj",
                  ]

PATH_MODEL_PRETRAIN_policy = "/ssdfs/datahome/tj20526/wx/LLM-Research/Llama-3___2-1B-Instruct"
REPO_ID_policy = "meta-llama/Llama-3.2-1B-Instruct"
PATH_MODEL_PRETRAIN_policy = PATH_MODEL_PRETRAIN_policy if PATH_MODEL_PRETRAIN_policy else REPO_ID_policy

PATH_MODEL_PRETRAIN_value = "/ssdfs/datahome/tj20526/wx/LLM-Research/Llama-3___2-1B-Instruct"
REPO_ID_value = "meta-llama/Llama-3.2-1B-Instruct"
PATH_MODEL_PRETRAIN_value = PATH_MODEL_PRETRAIN_value if PATH_MODEL_PRETRAIN_value else REPO_ID_value

DATA_PATH = "/root/autodl-tmp/autodl-tmp/llama3_sft/llama3_sft/dataset/advgen_toy.json"
MODEL_SAVE_DIR = "model_sft"

IS_PARALLELIZABLE = True
MODEL_PARALLEL = True
USE_CACHE = False
CUDA_VISIBLE_DEVICES = "0, 1"
USE_TORCH = "1"
CPU_NUMS = "8"
USE_CUDA = False if CUDA_VISIBLE_DEVICES == "-1" else True
USE_ALL_LOSS = True  # 计算loss时是否计算全部(False则为)

