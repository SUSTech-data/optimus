# %%

# %load_ext autoreload
# %autoreload 2
# %cd ../kaggle-llm
# %load_ext ipytorch
# %cluster start --n=4 --engines=MPI

# %%

from ipytorch import logging
logging.set_verbosity(logging.INFO)
# logging.set_verbosity(logging.DEBUG)
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
use_device = [0,1,4,5]
rank_device = dict(zip(range(world_size), use_device))
device = rank_device[rank]
torch.cuda.set_device(device)
mpu.initialize_model_parallel(4, topo)
deepspeed.checkpointing.configure(mpu, partition_activations=True)
mpu.model_parallel_cuda_manual_seed(seed)


# %%

from optimus.hf.llama import LlamaForCausalLM, LlamaConfig

config = LlamaConfig.from_pretrained("/data/hf/platypus-13b-mp4/part_0")
model = LlamaForCausalLM.from_pretrained(
    f"/data/hf/platypus-13b-mp4/part_{mpu.get_model_parallel_rank()}",
    config=config,
    torch_dtype=torch.bfloat16,
).eval().cuda()

# %%

question = "Introduce prime number"
header = f"""### Instruction:\n{question}"""
mid = "\n### Response:"
prompt = f"{header}{mid}"
print(prompt)

# %%

question = "Hello, who are you?"
header = f"""### Instruction:\n{question}"""
mid = "\n### Response:"
prompt1 = f"{header}{mid}"
print(prompt1)

# %%

from transformers import AutoTokenizer
import gc
tokenizer = AutoTokenizer.from_pretrained("/data/hf/platypus-13b-mp4/part_0")
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

# %%

model.llama.kv_enabled(True)
model.llama.checkpointing_enabled(False)

# %%


gc.collect()
torch.cuda.empty_cache()

inputs = tokenizer([prompt, prompt1], return_tensors="pt", max_length=64, padding="max_length")
# inputs = tokenizer(prompt, return_tensors="pt", max_length=64, padding=True)
# inputs = tokenizer(prompt, return_tensors="pt")
for k,v in inputs.items():
    inputs[k] = v.cuda()

inputs.input_ids

# %%

out = model.generate(**inputs, do_sample=False, max_length=100, left_padding=True, use_cache=True)
print("========================================")
print(tokenizer.decode(out[0].tolist()))
print("========================================")
print(tokenizer.decode(out[1].tolist()))
print("========================================")

# %%

model.llama.layers[0].attention.use_cache

