# %%

# %load_ext autoreload
# %autoreload 2
# %cd ../reasoning
# %load_ext ipytorch
# %cluster start --n=4 --engines=MPI

# %%

from ipytorch import logging

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

model.llama.fmha_enabled(False)
model.llama.kv_enabled(True)
# model.llama.fmha_enabled(True)
# model.llama.gradient_checkpointing = False

# %%

from datasets import load_dataset

gsm = load_dataset("gsm8k", "main", split="test")
gsm[1]


# %%

from transformers import AutoTokenizer, tokenization_utils_base
import gc

tokenizer = AutoTokenizer.from_pretrained("/data/hf/platypus-13b-mp4/part_0")
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

# %%

from transformers import DataCollatorWithPadding
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from optimus.generator.generator import Generator

torch.cuda.empty_cache()

def prompt_template(question):
    return f"### Instruction:\n\n{question}\n### Response:"
gsm_prompts = [prompt_template(q) for q in gsm["question"]]

# %%

generator = Generator(model, tokenizer, gsm_prompts, max_length=1024)
out = generator.run()

# %%

answers = gsm["answer"]
answers[:10]

# %%

import re

NUMERIC_IN_EN = r"(?:[\s=+-/<>($:\.\*\\])(?=\S)((?:0|(?:\d{1,3}(?:,\d{3})+(?=\D|$))|(?:\d+))(?:\.\d+)?%?)(?:(?![^\s=+-/>)$:\.\*\\])|(?=, ))"
NUMERIC_IN_ZH = (
    r"(?:\D|^)((?:0|(?:\d{1,3}(?:,\d{3})+(?=\D|$))|(?:\d+))(?:\.\d+)?%?)(?=\D|$)"
)


def extract_numeric(string, pattern=NUMERIC_IN_EN):
    # pattern = r"(?:[\s=+-/<>($:\.\*\\])(?=\S)((?:0|(?:[1-9](?:\d*|\d{0,2}(?:,\d{3})*)))?(?:\.\d+)?%?)(?:(?![^\s=+-/>)$:\.\*\\])|(?=, ))"
    try:
        all_values = list(
            filter(lambda x: len(x.strip()) != 0 and x != "%", re.findall(pattern, string))
        )

        def standardize(x):
            y = "".join(x.split(","))
            if "." in y:
                y = y.rstrip("0")
                if y[-1] == ".":
                    y = y[:-1]
            if y[0] == ".":
                y = "0" + y
            if y[-1] == "%":
                y = str(eval(y[:-1]) / 100)
            return y

        if not len(all_values):
            value = ""
        else:
            value = standardize(all_values[-1].strip())

        if value == "":
            return -9999.123
        else:
            return float(value)
    except Exception:
        return -9999.123

# %%

trues = list(map(extract_numeric, answers))
ans = list(map(extract_numeric, out))
right = 0
for i in range(len(trues)):
    # print(trues[i], ans[i])
    if trues[i] == ans[i]:
        right += 1
print(right / len(trues))


