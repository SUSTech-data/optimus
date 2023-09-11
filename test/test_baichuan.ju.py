# %%

# %load_ext autoreload
# %autoreload 2
# %cd ../reasoning
# %load_ext ipytorch
# %cluster start --n=4 --engines=MPI

# %%

from ipytorch import logging
from transformers import AutoTokenizer

logging.set_verbosity(logging.INFO)
import os

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import torch
import optimus.mpu as mpu
import deepspeed
import torch.distributed as dist
from deepspeed.runtime.pipe.topology import PipeModelDataParallelTopology

# %%

seed = 42
deepspeed.init_distributed(
    dist_backend="nccl",
    init_method="env://",
    verbose=False,
)
topo = PipeModelDataParallelTopology(
    num_pp=1, num_mp=4, num_dp=dist.get_world_size() // 4
)
rank = dist.get_rank()
world_size = dist.get_world_size()
use_device = [1, 2, 6, 7]
rank_device = dict(zip(range(world_size), use_device))
device = rank_device[rank]
torch.cuda.set_device(device)
mpu.initialize_model_parallel(4, topo)
deepspeed.checkpointing.configure(mpu, partition_activations=True)
mpu.model_parallel_cuda_manual_seed(seed)

# %%

from optimus.hf.llama import LlamaForCausalLM, LlamaConfig

config = LlamaConfig.from_pretrained("/data/hf/baichuan13b-mp4/part_0")
config.model_parallel_size = 4
model = (
    LlamaForCausalLM.from_pretrained(
        f"/data/hf/baichuan13b-mp4/part_{mpu.get_model_parallel_rank()}",
        config=config,
        torch_dtype=torch.bfloat16,
    )
    .eval()
    .cuda()
)

# %%

model.llama.kv_enabled(False)
model.llama.fmha_enabled(True)

# %%

def build_chat_input(tokenizer, messages: list[dict], max_new_tokens: int=2048):
    def _parse_messages(messages, split_role="user"):
        system, rounds = "", []
        round = []
        for i, message in enumerate(messages):
            if message["role"] == "system":
                assert i == 0
                system = message["content"]
                continue
            if message["role"] == split_role and round:
                rounds.append(round)
                round = []
            round.append(message)
        if round:
            rounds.append(round)
        return system, rounds

    model_max_length = 2048
    max_new_tokens = max_new_tokens or model.generation_config.max_new_tokens
    max_input_tokens = model_max_length - max_new_tokens
    system, rounds = _parse_messages(messages, split_role="user")
    system_tokens = tokenizer.encode(system)
    max_history_tokens = max_input_tokens - len(system_tokens)

    user_token_id = 195
    assistant_token_id = 196

    history_tokens = []
    for round in rounds[::-1]:
        round_tokens = []
        for message in round:
            if message["role"] == "user":
                round_tokens.append(user_token_id)
            else:
                round_tokens.append(assistant_token_id)
            round_tokens.extend(tokenizer.encode(message["content"]))
        if len(history_tokens) == 0 or len(history_tokens) + len(round_tokens) <= max_history_tokens:
            history_tokens = round_tokens + history_tokens  # concat left
            if len(history_tokens) < max_history_tokens:
                continue
        break

    input_tokens = system_tokens + history_tokens
    if messages[-1]["role"] != "assistant":
        input_tokens.append(assistant_token_id)
    input_tokens = input_tokens[-max_input_tokens:]  # truncate left
    return torch.LongTensor([input_tokens]).cuda()

# %%

tokenizer = AutoTokenizer.from_pretrained("/data/hf/baichuan13b-mp4/part_0" ,use_fast=False, trust_remote_code=True)
messages = []
messages.append({"role": "user", "content": "你是谁"})
tokens = build_chat_input(tokenizer, messages)

# %%

tokens

# %%

out = model.generate(input_ids=tokens, do_sample=False, max_length=256)

# %%

tokenizer.decode(out[0])

