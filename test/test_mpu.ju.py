# %%

# %load_ext autoreload
# %autoreload 2
# %cd ../kaggle-llm
# %load_ext ipytorch
# %cluster start --n=4 --engines=MPI

# %%

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
use_device = [4, 5, 6, 7]
rank_device = dict(zip(range(world_size), use_device))
device = rank_device[rank]
torch.cuda.set_device(device)
mpu.initialize_model_parallel(4, topo)
deepspeed.checkpointing.configure(mpu, partition_activations=True)
mpu.model_parallel_cuda_manual_seed(seed)

# %%

seed = 42
topo = PipeModelDataParallelTopology(
    num_pp=1, num_mp=2, num_dp=dist.get_world_size() // 2
)
rank = dist.get_rank()
world_size = dist.get_world_size()
use_device = [4, 5, 6, 7]
rank_device = dict(zip(range(world_size), use_device))
device = rank_device[rank]
torch.cuda.set_device(device)
mpu.initialize_model_parallel(2, topo)
deepspeed.checkpointing.configure(mpu, partition_activations=True)
mpu.model_parallel_cuda_manual_seed(seed)

# %%

from optimus.hf.llama import LlamaForCausalLM, LlamaConfig

config = LlamaConfig.from_pretrained("/data/hf/platypus-13b-mp4/part_0")
with mpu.scope(0):
    model = LlamaForCausalLM.from_pretrained(
        f"/data/hf/platypus-13b-mp4/part_{mpu.get_model_parallel_rank()}",
        config=config,
        torch_dtype=torch.bfloat16,
    ).eval().cuda()

with mpu.scope(1):
    model1 = LlamaForCausalLM.from_pretrained(
        f"/data/hf/platypus-13b-mp2/part_{mpu.get_model_parallel_rank()}",
        config=config,
        torch_dtype=torch.bfloat16,
    ).eval().cuda()

# %%

mpu(1)

if mpu.get_data_parallel_rank() == 0:
    question = "hello, who are you?"
    header = f"""### Instruction:\n\n{question}"""
    mid = "\n### Response:"
    prompt = f"{header}{mid}"
    print(prompt)

else:
    question = "What is the meaning of life?"
    header = f"""\### Instruction:\n\n{question}"""
    mid = "\n### Response:"
    prompt = f"{header}{mid}"
    print(prompt)



from transformers import AutoTokenizer
import gc


tokenizer = AutoTokenizer.from_pretrained("/data/hf/platypus-13b-mp4/part_0")
gc.collect()
torch.cuda.empty_cache()

inputs = tokenizer(prompt, return_tensors="pt")
for k,v in inputs.items():
    inputs[k] = v.cuda()

out = model1.generate(**inputs, do_sample=False, max_length=100)
print(tokenizer.decode(out[0].tolist()))

# %%

mpu(0)

if mpu.get_data_parallel_rank() == 0:
    question = "hello, who are you?"
    header = f"""\### Instruction:\n\n{question}"""
    mid = "\n### Response:"
    prompt = f"{header}{mid}"
    print(prompt)

else:
    question = "What is the meaning of life?"
    header = f"""\### Instruction:\n\n{question}"""
    mid = "\n### Response:"
    prompt = f"{header}{mid}"
    print(prompt)

from transformers import AutoTokenizer
import gc
tokenizer = AutoTokenizer.from_pretrained("/data/hf/platypus-13b-mp4/part_0")
gc.collect()
torch.cuda.empty_cache()

inputs = tokenizer(prompt, return_tensors="pt")
for k,v in inputs.items():
    inputs[k] = v.cuda()

out = model.generate(**inputs, do_sample=False, max_length=100)
print(tokenizer.decode(out[0].tolist()))
