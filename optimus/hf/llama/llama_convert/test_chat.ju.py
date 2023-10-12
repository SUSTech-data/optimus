# %%

# %load_ext autoreload
# %autoreload 2
# %cd ../kaggle-llm
# %load_ext ipytorch
# %cluster start --n=8 --engines=MPI

# %%

from ipytorch import logging
import gc
import os
import deepspeed
import torch
import torch.distributed as dist
from accelerate.utils import set_seed
from deepspeed.ops.adam import FusedAdam
from deepspeed.runtime.pipe.topology import PipeModelDataParallelTopology
# from megatron import mpu, fused_kernels
from torch.utils.data.distributed import DistributedSampler
from optimus.hf.llama import LlamaForCausalLM, LlamaForRM, LlamaConfig
import optimus.mpu as mpu
from optimus import load_fused_kernels
from transformers import LlamaTokenizer, DataCollatorWithPadding
from functools import wraps
from itertools import combinations
from typing import Iterable
from pathlib import Path
import pandas as pd
import socket
import numpy as np
from ipytorch.utils import is_notebook
import transformers
from tqdm.auto import tqdm

tqdm.pandas()
from functools import partial
import sys
from ipytorch.utils import is_notebook

# %%

if is_notebook():
    CHAT = True
    MP_SIZE = 4
    MODEL_DIR = Path("/data/hf/llama2-13b-chat-mp4")

else:
    import sys
    from absl import app, flags
    from absl.app import _run_init, parse_flags_with_usage

    FLAGS = flags.FLAGS

    flags.DEFINE_string(
        "model_path", None, "mp dst dir", short_name="m"
    )
    flags.DEFINE_boolean(
        "is_chat_model", True, "whether this Llama is a chat model", short_name="chat"
    )
    flags.DEFINE_integer(
        "model_parallel_size", 8, "model parallel size", short_name="p"
    )

    args = _run_init(sys.argv, parse_flags_with_usage)
    CHAT = FLAGS.is_chat_model
    MP_SIZE = FLAGS.model_parallel_size
    MODEL_DIR = Path(FLAGS.model_path)

# %%


def init_locally():
    seed = 42
    deepspeed.init_distributed(
        dist_backend="nccl",
        init_method="env://",
        verbose=False,
    )
    topo = PipeModelDataParallelTopology(
        num_pp=1, num_mp=MP_SIZE, num_dp=8//MP_SIZE
    )
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    use_device = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    # use_device = [1, 2, 4, 6]
    # use_device = [1, 2, 4, 5]
    rank_device = dict(zip(range(world_size), use_device))
    device = rank_device[rank]
    torch.cuda.set_device(device)
    mpu.initialize_model_parallel(MP_SIZE, topo)
    set_seed(seed)
    deepspeed.checkpointing.configure(mpu, partition_activations=True)
    mpu.model_parallel_cuda_manual_seed(seed)
    load_fused_kernels()

init_locally()

# %%

from transformers import AutoTokenizer
# dir = Path(f"/data/hf/llama2-70b-chat-mp8/part_{mpu.get_model_parallel_rank()}")
MODEL_DIR = MODEL_DIR / f"part_{mpu.get_model_parallel_rank()}"
config = LlamaConfig.from_pretrained(MODEL_DIR)
config.torch_dtype = "bfloat16"
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = LlamaForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=torch.bfloat16)
model.bfloat16().cuda().eval()

# %%


B_INST, E_INST = "[INST] ", " [/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""
MATH_SYSTEM_PROMPT = """\
You are an individual of exceptional logical capabilities, a master at deploying methodical and meticulous reasoning to solve problems. You excel particularly in solving complex mathematical equations, your mind effortlessly tracing a clear and logical path to the solution. To you, numbers and formulas are akin to an intricate maze, and yet you always manage to pinpoint the path leading to the correct solution swiftly and accurately.
Your thought process is agile, precise, and disciplined, as if operating on its own sophisticated algorithm, making you outstanding in the field of mathematics. But more notably, your approach to problem-solving is characterized by a step-by-step process, akin to methodically climbing a staircase. Every mathematical problem you encounter is a new challenge, and you carefully dissect each problem into manageable steps, deducing and progressing logically from one step to the next.
Your deep understanding of mathematics and your precise grasp of logical structures allow you to dance elegantly and accurately through the rhythm of numbers. This systematic, sequential, and orderly thought process leads you not just to the answer, but the comprehension of the journey to the answer."""

# %%

if CHAT:
    # question = "What is the definition of a prime number?"
    question = "hello, who are you?"
    prompt = B_SYS + DEFAULT_SYSTEM_PROMPT + E_SYS + B_INST + question + E_INST
    max_length = 400
else:
    prompt = "I will introduce Beijing, Beijing is a city"
    max_length = 200


tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side

inputs = tokenizer(prompt, return_tensors="pt")
inputs

# %%

model.llama.fmha_enabled(True)

# %%

outputs = model.generate(inputs["input_ids"].cuda(), attention_mask=inputs["attention_mask"].cuda(), max_length=max_length)
outputs

# %%

if dist.get_rank() == 0:
    print(tokenizer.decode(outputs[0]))

dist.barrier()

