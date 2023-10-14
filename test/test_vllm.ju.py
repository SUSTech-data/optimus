# %%

# %load_ext autoreload
# %autoreload 2
# %cd ../optimus/test
# %load_ext ipytorch
# %cluster start --n=4 --engines=MPI

# %%

from ipytorch import logging
import os

logging.set_verbosity(logging.INFO)

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,4,5"

import torch
from optimus import mpu
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
device = rank
torch.cuda.set_device(device)
mpu.initialize_model_parallel(4, topo)
deepspeed.checkpointing.configure(mpu, partition_activations=True)
mpu.model_parallel_cuda_manual_seed(seed)

# %%

from optimus.hf.llama import LlamaForCausalLM, LlamaConfig

config = LlamaConfig.from_pretrained("/data/hf/platypus-13b-mp4/part_0")
model = (
    LlamaForCausalLM.from_pretrained(
        f"/data/hf/platypus-13b-mp4/part_{mpu.get_model_parallel_rank()}",
        config=config,
        torch_dtype=torch.bfloat16,
    )
    .eval()
    .cuda()
)

# %%

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("/data/hf/platpus")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = "left"

# %%

data = tokenizer("### Instruction:\nhello, who are you\n### Response:", return_tensors="pt")
data = data.to("cuda")

# %%

model.llama.kv_enabled(True)
model.llama.fmha_enabled(False)

# %%

model.generate(**data,do_sample=False)

# %%

import argparse
import json
import random
import time
from typing import List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, PreTrainedTokenizerBase
from tqdm import tqdm

from vllm import LLM, SamplingParams
from vllm.transformers_utils.tokenizer import get_tokenizer
from optimus.utils import release_cuda

model.cpu()
release_cuda()

# %%

llm = LLM(
    model="/data/hf/platpus",
    tokenizer="/data/hf/platpus",
    tensor_parallel_size=1,
    optimus_config=config,
    optimus_worker="llama",
    dtype="bfloat16",
    state_dict=model.state_dict(),
)

# %%

from datasets import load_dataset
ds = load_dataset("gsm8k","main")
ds

# %%

questions = [d["question"] for d in ds["test"]]
prompts = [f"### Instruction:\n{q}\n### Response:" for q in questions[:32]]
requests = [(p, None, 1024) for p in prompts]

# %%

use_beam_search = False
n=1
for prompt, _, output_len in requests:
    sampling_params = SamplingParams(
        n=n,
        temperature=0.0 if use_beam_search else 1.0,
        top_p=1.0,
        use_beam_search=use_beam_search,
        ignore_eos=False,
        max_tokens=output_len,
    )
    # FIXME(woosuk): Do not use internal method.
    llm._add_request(
        prompt=prompt,
        prompt_token_ids=None,
        sampling_params=sampling_params,
    )

# %%

out = llm._run_engine(use_tqdm=True)
out

# %%

out[0]
