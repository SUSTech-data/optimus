# %%

import sys

import psutil
from absl import logging

sys.executable
logging.set_verbosity(logging.INFO)
logging.debug("Hello")

# %%

# %load_ext ipytorch

# %%

# %cluster start --n=4 --engines=MPI

# %%

"""
# %nopx
# %cluster restart
# logging.info("hello")
"""

# %%

# %load_ext autoreload
# %autoreload 2

# %%

# device_list = [1,2,4,5]

import gc
import os
from collections import deque

import deepspeed
import torch
import torch.distributed as dist
from accelerate.utils import set_seed
from deepspeed.ops.adam import FusedAdam
from deepspeed.runtime.pipe.topology import PipeModelDataParallelTopology
from fengshen_inner.models.llama.modeling_llama import *
from fengshen_inner.models.megatron import fused_kernels, mpu

from chatgpt.utils.utils import release_cuda

# %%


seed = 42
deepspeed.init_distributed(
    dist_backend="nccl",
    init_method="env://",
    verbose=False,
)
topo = PipeModelDataParallelTopology(num_pp=1, num_mp=4, num_dp=1)
rank = dist.get_rank()
world_size = dist.get_world_size()
# use_device = [0, 1, 3, 4]
# use_device = [1, 2, 4, 6]
use_device = [1, 2, 4, 5]
rank_device = dict(zip(range(world_size), use_device))
device = rank_device[rank]
torch.cuda.set_device(device)
mpu.initialize_model_parallel(4, topo)
set_seed(seed)
deepspeed.checkpointing.configure(mpu, partition_activations=True)
mpu.model_parallel_cuda_manual_seed(seed)
fused_kernels.load_fused_kernels()

# %%

model_path = f"/data/hf/llama2-13b-chat-mp4/part_{mpu.get_model_parallel_rank()}"
model = LlamaForCausalLM.from_pretrained(model_path).half().cuda()
# release_cuda()

# %%

model.config.use_cache

# %%

from transformers import AutoTokenizer

tokenizer_path = "/home/rok/llama/llama2-13b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

# %%

import gc

gc.collect()
torch.cuda.empty_cache()

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

question = "Hello, who are you?"
prompt = B_SYS + DEFAULT_SYSTEM_PROMPT + E_SYS + B_INST + question + E_INST
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side

# %%

# import tree

# inputs = tokenizer(
#     prompt, return_tensors=None, padding="max_length", truncation=False, max_length=256
# )
# inputs = tree.map_structure_up_to(
#     {"input_ids": None, "attention_mask": None},
#     lambda x: torch.tensor(x, dtype=torch.long, device="cuda").reshape(1, -1),
#     inputs,
# )
# inputs

# # %%

# input_ids = inputs["input_ids"]
# attention_mask = inputs["attention_mask"]
# position_ids = attention_mask.long().cumsum(-1) - 1
# position_ids.masked_fill_(attention_mask == 0, 1)

# # %%

# model.eval()
# with torch.no_grad():
#     out = model(
#         input_ids=input_ids,
#         attention_mask=attention_mask,
#         position_ids=position_ids,
#         use_cache=True,
#     )
#     # out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)


# kv = out.past_key_values
# new_kv = []
# for lkv in kv:
#     new_lkv = lkv.clone()
#     new_lkv[0, :10] = 1.8
#     new_lkv[1, :10] = 1.8
#     new_kv.append(new_lkv)
# # new_kv.shape

# # %%

# next_token = torch.argmax(out.logits[0, -1, :], dim=-1).reshape(1, 1)
# input_ids = torch.cat([input_ids, next_token], dim=1)
# attention_mask = torch.cat([attention_mask, torch.ones(1, 1, device="cuda")], dim=1)
# print(tokenizer.decode(input_ids[0][-1].tolist()))


# def prepare(input_ids, attention_mask):
#     # input_ids = inputs["input_ids"]
#     # attention_mask = inputs["attention_mask"]
#     input_ids = input_ids[:, -1:]
#     position_ids = attention_mask.long().cumsum(-1) - 1
#     position_ids.masked_fill_(attention_mask == 0, 1) position_ids = position_ids[:, -1].unsqueeze(-1)
#     return {
#         "input_ids": input_ids,
#         "attention_mask": attention_mask,
#         "position_ids": position_ids,
#         "use_cache": True,
#     }


# inputs = prepare(input_ids, attention_mask)
# with torch.no_grad():
#     out = model(past_key_values=new_kv, **inputs)
#     kv = out.past_key_values
#     new_kv = []
#     for lkv in kv:
#         new_lkv = lkv.clone()
#         new_lkv[0, :10] = 0.0
#         new_lkv[1, :10] = 1.0
#         new_kv.append(new_lkv)


# %%

from datasets import load_dataset

gsm_dataset = load_dataset("gsm8k", "main", split="train")

# %%

from chatgpt.utils import extract_numeric

questions = gsm_dataset["question"][:256]
answers = [extract_numeric(a) for a in gsm_dataset["answer"][:128]]
# answers

# %%

import tree
from ipytorch import logging
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from chatgpt.utils import create_device_collator

logging.set_verbosity(logging.WARNING)

# %%

prompts = [
    B_SYS
    + DEFAULT_SYSTEM_PROMPT
    + E_SYS
    + B_INST
    + "This is a math problem, please think step by step to solve it: "
    + q
    + E_INST
    for q in questions
]
prompts[0]

# %%

import tree

class Sequence:  # note that this object only exists on stage 1 and elder
    def __init__(self, input_ids, attention_mask, global_idx):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        # self.kv = kv  # kv is stacked as [num_layers, 2, seq_len, `ignore batch_size`, num_heads, head_dim]
        # self.move_event = torch.cuda.Event()
        self.global_idx = global_idx

    @property
    def shape(self):
        return self.input_ids.shape

    """
    `cuda` and `cpu` should be ensured to exec
    at some cuda stream like `kv_stream`
    """

    def cuda(self):
        self.kv = self.kv.to("cuda", non_blocking=True)
        self.move_event.record()
        return self

    def cpu(self):
        self.kv = self.kv.to("cpu", non_blocking=True)
        self.move_event.record()
        return self

    def wait(self, stream=None):
        if stream is None:
            self.move_event.wait()
        else:
            self.move_event.wait(stream)

    def __len__(self):
        return self.shape[-1]


# %%


class GenerationManager:
    def __init__(self, model, tokenizer, prompts, stages=None):
        # self.model = model.eval()
        model.eval()
        self.model = model
        self.prompts = prompts
        self.prompt_ptr = 0

        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        self.eos_token_id = tokenizer.eos_token_id

        """
        Stage is a list of tuple of (LENGTH, THRESHOULD, GENERATION_BATCH_SIZE)
        when batch of sequence reach THRESHOULD, this data manager
        would cut the sequence whose length > LENGTH,
        and `trim` the batch information, give new sequence into batch,
        then model will inference
        """
        # _default_stages = [(3, 4), (7, 8), (11, 12)]
        # _default_stages = tree.map_structure(lambda x: 128 * x, _default_stages)
        _default_stages = [(400, 450, 64), (550, 600, 48), (700, 750, 32)]
        self.stages = stages if stages is not None else _default_stages
        self.stage_ptr = 0

        self.stage_buffer: deque[Sequence] = deque()

        """
        Generation runtime information
        """
        self.input_ids = None
        self.attention_mask = None
        self.kv = None

        """
        Output information
        """
        self.batch_global_idxs = (
            []
        )  # dynamic changing, whose length always equal to len(self.input_ids)

        self.outputs = []
        self.output_idxs = []  # global idxs

    def _get_from_stage_buffer(self, size) -> list[Sequence]:
        out = [self.stage_buffer.popleft() for _ in range(size)]
        return out

    def _get_buffer_sequences_and_fill_state(self, size, padding_length):
        # this function still need padding size
        # since during generating tokens from 512 -> 768,
        # there would be some sentences finished,
        # then we need to pad and push (input_ids, attention_mask, kv) to the running batch
        pad_token_id = self.pad_token_id
        seqs: list(Sequence) = self._get_from_stage_buffer(size)
        input_ids_list = []
        attention_mask_list = []
        for seq in seqs:
            input_ids_list.append(seq.input_ids)
            attention_mask_list.append(seq.attention_mask)
            self.batch_global_idxs.append(seq.global_idx)

        input_ids = torch.stack(input_ids_list, dim=0)
        attention_mask = torch.stack(attention_mask_list, dim=0)

        if padding_length is not None:
            input_ids = torch.nn.functional.pad(
                input_ids, (padding_length, 0), value=pad_token_id
            )
            attention_mask = torch.nn.functional.pad(
                attention_mask, (padding_length, 0), value=0
            )
            # pad to `padding_length` got `padding_length+1`

        # forward pass new sentences once to get kv
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        out = self.forward_pass(
            input_ids, attention_mask, position_ids, use_own_kv=False
        )
        kv = out.past_key_values
        next_token_logits = out.logits[:, -1, :]
        next_tokens = torch.argmax(next_token_logits, dim=-1)
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        attention_mask = torch.cat(
            [
                attention_mask,
                attention_mask.new_ones((attention_mask.shape[0], 1)),
            ],
            dim=-1,
        )

        # concat to self states
        self.input_ids = torch.cat([self.input_ids, input_ids], dim=0)
        self.attention_mask = torch.cat([self.attention_mask, attention_mask], dim=0)
        for i in range(len(self.kv)):
            self.kv[i] = torch.cat([self.kv[i], kv[i]], dim=2)

    @property
    def run_shape(self):
        return self.input_ids.shape

    @property
    def run_batch_size(self):
        return self.run_shape[0]  # note that this property can be 0

    @property
    def run_seq_len(self):
        return self.run_shape[1]

    def _get_prompts(self, size, padding_to):
        """
        StopIteration would be raised here
        This is a `proxy function` for different backend
        of `self.prompts` object and this interface should
        have same input signature of `size` and `padding length`
        """
        start = self.prompt_ptr
        end = self.prompt_ptr + size
        if end > len(self.prompts):
            raise StopIteration

        end = end if end <= len(self.prompts) else len(self.prompts)
        selected = self.prompts[start:end]
        idxs = range(start, end)
        self.prompt_ptr = end
        self.batch_global_idxs.extend(
            idxs
        )  # extend since next sentence is always been concated on the end of batch

        if padding_to is None:
            inputs = tokenizer(
                selected, return_tensors=None, padding="longest", truncation=False
            )
        else:
            inputs = tokenizer(
                selected,
                return_tensors=None,
                padding="max_length",
                truncation=True,
                max_length=padding_to,
            )
        inputs = tree.map_structure_up_to(
            {"input_ids": None, "attention_mask": None},
            lambda x: torch.tensor(x, dtype=torch.long, device="cuda").reshape(
                size, -1
            ),
            inputs,
        )
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        return input_ids, attention_mask

    def prepare_inputs_for_forward(self):
        if self.kv is not None:
            input_ids = self.input_ids[:, -1:]
            attention_mask = self.attention_mask
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = position_ids[:, -1].unsqueeze(-1)
            return input_ids, attention_mask, position_ids
        else:
            input_ids = self.input_ids
            attention_mask = self.attention_mask
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            return input_ids, attention_mask, position_ids

    @torch.no_grad()
    def forward_pass(self, input_ids, attention_mask, position_ids, use_own_kv=True):
        try:
            if use_own_kv:
                out = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=self.kv,
                    use_cache=True,
                )
            else:
                out = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    use_cache=True,
                )
        except Exception as e:
            logging.error(f"Error: {e}")
            logging.error(f"Input ids: {input_ids.shape}")
            logging.error(f"Attention mask: {attention_mask.shape}")
            logging.error(f"Position ids: {position_ids.shape}")
            logging.error(f"kv: {len(self.kv)} with shape: {self.kv[0].shape}")
            raise e
        return out

    def _greedy_decode(self, out):
        next_token_logits = out.logits[:, -1, :]
        next_tokens = torch.argmax(next_token_logits, dim=-1)
        self.input_ids = torch.cat([self.input_ids, next_tokens[:, None]], dim=-1)
        self.attention_mask = torch.cat(
            [
                self.attention_mask,
                self.attention_mask.new_ones((self.attention_mask.shape[0], 1)),
            ],
            dim=-1,
        )
        self.kv = out.past_key_values
        return next_tokens

    def _need_trim(self):
        try:
            min_padding = min(
                (self.attention_mask != 0).to(dtype=torch.int8).argmax(dim=1)
            )
        except ValueError:
            return False, 0
        need_trim = min_padding > 0
        return need_trim, min_padding

    def _trim(self, min_padding, unfinished_mask=None):
        # unfinished mask is only used for kv cache to fuse cut operation
        self.input_ids = self.input_ids[:, min_padding:]
        self.attention_mask = self.attention_mask[:, min_padding:]
        if unfinished_mask is not None:
            for i in range(len(self.kv)):
                self.kv[i] = self.kv[i][:, min_padding:, unfinished_mask, :, :]
        else:
            for i in range(len(self.kv)):
                self.kv[i] = self.kv[i][:, min_padding:]
        logging.info(
            f"Trim outputs: {self.input_ids.shape}, {self.attention_mask.shape}, {self.kv[0].shape} with {min_padding}"
        )

    # create this function to avoid `out` as a mid param to use cuda mem
    @torch.no_grad()
    def forward_pass_with_greedy_decode(
        self, input_ids, attention_mask, position_ids, use_own_kv=True
    ):
        out = self.forward_pass(input_ids, attention_mask, position_ids, use_own_kv)
        return self._greedy_decode(out)

    def process_eos(self, next_tokens):
        eos = self.eos_token_id
        this_token_eos = (next_tokens == eos).squeeze()
        num_finished = this_token_eos.sum()
        if num_finished > 0:
            unfinished = ~this_token_eos
            finished = self.input_ids[this_token_eos]
            finished_batch_idxs = torch.nonzero(this_token_eos).squeeze().tolist()
            if not isinstance(finished_batch_idxs, list):
                finished_batch_idxs = [finished_batch_idxs]
            finished_global_idxs = [
                self.batch_global_idxs[k] for k in finished_batch_idxs
            ]

            self.outputs.append(finished)
            self.output_idxs.append(finished_global_idxs)

            # update self states
            self.batch_global_idxs = [
                p
                for i, p in enumerate(self.batch_global_idxs)
                if i not in finished_batch_idxs
            ]

            self.input_ids = self.input_ids[unfinished]
            self.attention_mask = self.attention_mask[unfinished]

            # fused two dims of kv trim into one operation
            # self.kv = [lkv[:, :, unfinished, :, :] for lkv in self.kv]
            # self.kv = [lkv[:, min_padding: ] for lkv in self.kv]

            need_trim, min_padding = self._need_trim()
            if need_trim:
                self._trim(min_padding, unfinished_mask=unfinished)
            else:
                for i in range(len(self.kv)):
                    self.kv[i] = self.kv[i][:, :, unfinished, :, :]

    def cut_sentence(self, length, threshold):
        # Note that the input of this funciton has been trimmed
        # and eos has been processed, i.e. no finished senteces
        LONG_SENTENCE = length
        LONG_SENTENCE_THRESHOLD = threshold
        length = self.run_seq_len
        if length >= LONG_SENTENCE_THRESHOLD:
            logging.info(f"Length {length} of threshold {LONG_SENTENCE_THRESHOLD}")
            sentence_length = self.attention_mask.sum(dim=1)
            long_sentence_mask = (sentence_length > LONG_SENTENCE).squeeze()
            # long_input_ids = self.input_ids[long_sentence_mask]
            # long_attention_mask = self.attention_mask[long_sentence_mask]
            # num_long_sentences = len(long_input_ids)

            long_sentence_batch_idxs = (
                torch.nonzero(long_sentence_mask).squeeze().tolist()
            )
            if not isinstance(long_sentence_batch_idxs, list):
                long_sentence_batch_idxs = [long_sentence_batch_idxs]
            logging.info(
                f"Cut {len(long_sentence_batch_idxs)} sentence with length {length}"
            )
            long_sentence_global_idxs = [
                self.batch_global_idxs[i] for i in long_sentence_batch_idxs
            ]

            for i in long_sentence_batch_idxs:
                sequence = Sequence(
                    self.input_ids[i],
                    self.attention_mask[i],
                    self.batch_global_idxs[i],
                )
                self.stage_buffer.append(sequence)

            self.batch_global_idxs = [
                p
                for i, p in enumerate(self.batch_global_idxs)
                if i not in long_sentence_batch_idxs
            ]
            unfinished_mask = ~long_sentence_mask
            self.input_ids = self.input_ids[unfinished_mask]
            self.attention_mask = self.attention_mask[unfinished_mask]
            need_trim, min_padding = self._need_trim()
            if need_trim:
                self._trim(min_padding, unfinished_mask)
            else:
                for i in range(len(self.kv)):
                    self.kv[i] = self.kv[i][:, :, unfinished_mask, :, :]

    def run_init_stage(self):
        # use all prompts in self.prompts
        # generating until stage1, i.e. (384, 512)

        LONG_SENTECE_LENGTH, THRESHOLD, mini_batch_size = self.stages[0]
        # sentence on init stage that has been sent to stage buffer have length 512

        self.input_ids, self.attention_mask = self._get_prompts(mini_batch_size, None)
        input_ids, attention_mask, position_ids = self.prepare_inputs_for_forward()
        # assume that first token is not eos
        self.forward_pass_with_greedy_decode(
            input_ids, attention_mask, position_ids
        )  # use `self.kv`

        # enter core loop of stage 0
        while True:
            # 1. prepare kv cache and position_ids
            input_ids, attention_mask, position_ids = self.prepare_inputs_for_forward()

            # 2. forward pass and greedy decode
            next_tokens = self.forward_pass_with_greedy_decode(
                input_ids, attention_mask, position_ids
            )

            # 3. process eos of output tokens to update `input_ids`, `attention_mask`, `kv`
            self.process_eos(next_tokens)

            # 4. cut long sentence and send them to `self.stage_buffer` as `Sequence` object
            self.cut_sentence(LONG_SENTECE_LENGTH, THRESHOLD)

            # 5. get new sentences
            run_batch_size, run_seq_len = self.run_shape
            need_size = mini_batch_size - run_batch_size
            if need_size > 0:
                try:
                    input_ids, attention_mask = self._get_prompts(
                        need_size, run_seq_len - 1
                    )  # forward pass once to get kv
                except StopIteration:
                    break

                # 6. forward pass new sentences once to get kv
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                out = self.forward_pass(
                    input_ids, attention_mask, position_ids, use_own_kv=False
                )
                kv = out.past_key_values
                next_token_logits = out.logits[:, -1, :]
                next_tokens = torch.argmax(next_token_logits, dim=-1)
                input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
                attention_mask = torch.cat(
                    [
                        attention_mask,
                        attention_mask.new_ones((attention_mask.shape[0], 1)),
                    ],
                    dim=-1,
                )

                # 7. concat to self states
                self.input_ids = torch.cat([self.input_ids, input_ids], dim=0)
                self.attention_mask = torch.cat(
                    [self.attention_mask, attention_mask], dim=0
                )
                for i in range(len(self.kv)):
                    self.kv[i] = torch.cat([self.kv[i], kv[i]], dim=2)

        # 8. generate to next stage basic length
        next_stage_length = THRESHOLD
        while self.run_seq_len < (next_stage_length + 1) and self.run_batch_size > 0:
            input_ids, attention_mask, position_ids = self.prepare_inputs_for_forward()
            next_tokens = self.forward_pass_with_greedy_decode(
                input_ids, attention_mask, position_ids
            )
            self.process_eos(next_tokens)

    def run_stage(self, i):
        if i != len(self.stages) - 1:  # not final stage
            # last_stage_capicity = len(self.stage_prefetch_buffer) + len(self.stage_buffer)
            last_stage_capicity = len(self.stage_buffer)
            last_stage_seq_len = self.stages[i][1]  # threshould
            num_picking_from_buffer = 0
            LONG_SENTECE_LENGTH, THRESHOLD, mini_batch_size = self.stages[i + 1]
            logging.warning(
                f"Running stage {i} with capacity {last_stage_capicity} of length {last_stage_seq_len}"
            )

            while True:
                # 1. fill self state
                run_batch_size, run_seq_len = self.run_shape
                num_want = mini_batch_size - run_batch_size
                if num_want > 0:
                    buffer_left = last_stage_capicity - num_picking_from_buffer
                    pick_size = min(buffer_left, num_want)
                    if num_picking_from_buffer >= last_stage_capicity:
                        # no more sentence in buffer
                        break
                    num_picking_from_buffer += pick_size
                    padding_length = run_seq_len - last_stage_seq_len - 1
                    padding_length = padding_length if padding_length > 0 else None
                    self._get_buffer_sequences_and_fill_state(pick_size, padding_length)

                # 1. prepare kv cache and position_ids
                (
                    input_ids,
                    attention_mask,
                    position_ids,
                ) = self.prepare_inputs_for_forward()

                # 2. forward pass and greedy decode
                next_tokens = self.forward_pass_with_greedy_decode(
                    input_ids, attention_mask, position_ids
                )

                # 3. process eos of output tokens to update `input_ids`, `attention_mask`, `kv`
                self.process_eos(next_tokens)

                # 4. cut long sentence and send them to `self.stage_buffer` as `Sequence` object
                self.cut_sentence(LONG_SENTECE_LENGTH, THRESHOLD)

            next_stage_length = THRESHOLD
            while (
                self.run_seq_len < (next_stage_length + 1) and self.run_batch_size > 0
            ):
                (
                    input_ids,
                    attention_mask,
                    position_ids,
                ) = self.prepare_inputs_for_forward()
                next_tokens = self.forward_pass_with_greedy_decode(
                    input_ids, attention_mask, position_ids
                )
                self.process_eos(next_tokens)

        else:  # final stage
            last_stage_capicity = len(self.stage_buffer)
            last_stage_seq_len = self.stages[i][1]  # threshould
            num_picking_from_buffer = 0
            mini_batch_size = 32
            while True:
                # 1. fill self state
                run_batch_size, run_seq_len = self.run_shape
                num_want = mini_batch_size - run_batch_size
                if num_want > 0:
                    buffer_left = last_stage_capicity - num_picking_from_buffer
                    pick_size = min(buffer_left, num_want)
                    if num_picking_from_buffer >= last_stage_capicity:
                        # no more sentence in buffer
                        break
                    num_picking_from_buffer += pick_size
                    padding_length = run_seq_len - last_stage_seq_len - 1
                    padding_length = padding_length if padding_length > 0 else None
                    self._get_buffer_sequences_and_fill_state(pick_size, padding_length)

                # 1. prepare kv cache and position_ids
                (
                    input_ids,
                    attention_mask,
                    position_ids,
                ) = self.prepare_inputs_for_forward()

                # 2. forward pass and greedy decode
                next_tokens = self.forward_pass_with_greedy_decode(
                    input_ids, attention_mask, position_ids
                )

                # 3. process eos of output tokens to update `input_ids`, `attention_mask`, `kv`
                self.process_eos(next_tokens)

            while self.run_batch_size > 0:
                (
                    input_ids,
                    attention_mask,
                    position_ids,
                ) = self.prepare_inputs_for_forward()
                next_tokens = self.forward_pass_with_greedy_decode(
                    input_ids, attention_mask, position_ids
                )
                # logging.info(f"BATCH GLOBAL IDXS {self.batch_global_idxs}")
                self.process_eos(next_tokens)

    def get_output(self):
        return self.outputs, self.output_idxs


# %%

manager = GenerationManager(model, tokenizer, prompts)
manager.run_init_stage()
manager.run_stage(0)
manager.run_stage(1)
manager.run_stage(2)

# %%

out, idxs = manager.get_output()

# %%

a = []
b = []
for i in range(len(out)):
    a.extend(out[i])
    b.extend(idxs[i])

logging.info(len(a))
logging.info(len(b))

# %%

b[-1]

# %%

import time

from transformers import GenerationConfig

logging.set_verbosity(logging.INFO)

manager = GenerationManager(model, tokenizer, prompts)
config = GenerationConfig(do_sample=False, max_length=2048)
out = []
for i in range(4):
    start = time.time()
    logging.info(f"BATCH {i+1} / 4 start")
    input_ids, attention_mask = manager._get_prompts(64, None)
    out.append(
        model.generate(
            input_ids, attention_mask=attention_mask, generation_config=config
        )
    )

    end = time.time()
    logging.info(f"BATCH {i+1} / 4 end, estimated time: {end-start:.3f} seconds")

# %%



# tokenizer.batch_decode(a)

# # %%

# prompts[21]

# # %%


# class DataGenerator:
#     def __init__(self, tokenizer, prompts) -> None:
#         self.tokenizer = tokenizer
#         self.prompts = prompts
#         self.long_sentence_buffer = []

#         self.prompt_ptr = 0
#         self.long_sentence_ptr = 0

#     def _get_prompts(self, size, padding_length=None):
#         # assert remaining is enough for `size`
#         selected = self.prompts[self.prompt_ptr : self.prompt_ptr + size]
#         self.prompt_ptr += size

#         if padding_length is None:
#             inputs = tokenizer(
#                 selected, return_tensors=None, padding="longest", truncation=False
#             )
#         else:
#             inputs = tokenizer(
#                 selected,
#                 return_tensors=None,
#                 padding="max_length",
#                 truncation=True,
#                 max_length=padding_length,
#             )
#         inputs = tree.map_structure_up_to(
#             {"input_ids": None, "attention_mask": None},
#             lambda x: torch.tensor(x, dtype=torch.long, device="cuda").reshape(
#                 size, -1
#             ),
#             inputs,
#         )
#         input_ids = inputs["input_ids"]
#         attention_mask = inputs["attention_mask"]
#         return input_ids, attention_mask

#     def _get_long_sentence(self, size, padding_length):
#         selected = self.long_sentence_buffer[
#             self.long_sentence_ptr : self.long_sentence_ptr + size
#         ]
#         self.long_sentence_ptr += size
#         pad_token_id = self.tokenizer.pad_token_id

#         length = max(map(lambda x: len(x[0]), selected))
#         if padding_length is not None:
#             length = max(length, padding_length)

#         def _pad_to_length(input_ids, attention_mask):
#             shape = input_ids.shape
#             seq_len = shape[-1]
#             pad_length = length - seq_len
#             if pad_length > 0:
#                 input_ids = torch.nn.functional.pad(
#                     input_ids, (pad_length, 0), value=pad_token_id
#                 )
#                 attention_mask = torch.nn.functional.pad(
#                     attention_mask, (pad_length, 0), value=0
#                 )
#             return input_ids, attention_mask

#         input_ids_list = []
#         attention_mask_list = []
#         for input_ids, attention_mask in selected:
#             _input_ids, _attention_mask = _pad_to_length(input_ids, attention_mask)
#             input_ids_list.append(_input_ids)
#             attention_mask_list.append(_attention_mask)
#         input_ids = torch.stack(input_ids_list, dim=0)
#         attention_mask = torch.stack(attention_mask_list, dim=0)
#         return input_ids, attention_mask

#     @property
#     def num_prompt_remaining(self):
#         return len(self.prompts) - self.prompt_ptr

#     @property
#     def num_long_sentences_remaining(self):
#         return len(self.long_sentence_buffer) - self.long_sentence_ptr

#     @property
#     def num_remaining(self):
#         return self.num_prompt_remaining + self.num_long_sentences_remaining

#     def get_prompts(self, size, padding_length=None):
#         prompt_rem = self.num_prompt_remaining

#         if prompt_rem == 0:
#             raise StopIteration

#         if size <= prompt_rem:
#             return self._get_prompts(size, padding_length)

#         else:
#             prompt_size = size - prompt_rem
#             return self._get_prompts(prompt_size, padding_length)

#     def get_long_sentences(self, size, padding_length=None):
#         sentence_rem = self.num_long_sentences_remaining
#         if sentence_rem == 0:
#             raise StopIteration

#         if size <= sentence_rem:
#             return self._get_long_sentence(size, padding_length)

#         else:
#             sentence_size = size - sentence_rem
#             return self._get_long_sentence(sentence_size, padding_length)

#     def send(self, *args, **kwargs):
#         return self.get_prompts(*args, **kwargs)


# # %%

# # next(data_generator)

# # %%

# """
# KV shape: [2, seq_len, batch_size, num_head, head_size] with mp head size
# """


# @torch.no_grad()
# def forward_from_inputs(model, input_ids, attention_mask, kv=None, position_ids=None):
#     if position_ids is None:
#         position_ids = attention_mask.long().cumsum(-1) - 1
#         position_ids.masked_fill_(attention_mask == 0, 1)
#     try:
#         out = model(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             position_ids=position_ids,
#             past_key_values=kv,
#             use_cache=True,
#         )
#     except Exception as e:
#         logging.error(f"Error: {e}")
#         logging.error(f"Input ids: {input_ids.shape}")
#         logging.error(f"Attention mask: {attention_mask.shape}")
#         logging.error(f"Position ids: {position_ids.shape}")
#         logging.error(f"kv: {len(kv)} with shape: {kv[0].shape}")
#         raise e
#     return out


# @torch.no_grad()
# def concat_kv_on_batch(kv, new_kv):
#     logging.info(f"concat kv on batch with kv {kv[0].shape} and {new_kv[0].shape}")
#     assert len(kv) == len(new_kv)
#     # new = []
#     # for lkv, _lkv in zip(kv, new_kv):
#     #     new_lkv = torch.cat([lkv, _lkv], dim=2)
#     #     new.append(new_lkv)
#     for i in range(len(kv)):
#         kv[i] = torch.cat([kv[i], new_kv[i]], dim=2)
#     logging.info(f"concated kv {kv[0].shape}")
#     return kv


# @torch.no_grad()
# def trim_inputs(input_ids, attention_mask, kv=None):
#     logging.info(f"Trim inputs: {input_ids.shape}, {attention_mask.shape}")
#     # 找到每个序列的第一个非零元素的位置
#     min_padding = min((attention_mask != 0).to(dtype=torch.int8).argmax(dim=1))
#     need_trim = min_padding > 0
#     if need_trim:
#         # 使用Python切片语法裁剪序列
#         input_ids = input_ids[:, min_padding:]
#         attention_mask = attention_mask[:, min_padding:]
#         if kv is not None:
#             for i in range(len(kv)):
#                 kv[i] = kv[i][:, min_padding:]
#         logging.info(
#             f"Trim outputs: {input_ids.shape}, {attention_mask.shape}, {kv[0].shape} with {min_padding}"
#         )
#     else:
#         logging.info(f"No need to trim")
#     return input_ids, attention_mask, kv


# def cut_long_sentence(
#     input_ids, attention_mask, kv, long_sentence_buffer, eos_token_id
# ):
#     """
#     Note that the input of this funciton has been trimmed
#     """
#     LONG_SENTENCE = 600
#     LONG_SENTENCE_THRESHOLD = 650
#     # input_ids = input_ids[:, -1:]
#     length = input_ids.shape[1]
#     if length > LONG_SENTENCE_THRESHOLD:
#         sentence_length = attention_mask.sum(dim=1)
#         long_sentence_mask = (
#             (sentence_length > LONG_SENTENCE) & (input_ids[:, -1] != eos_token_id)
#         ).squeeze()
#         long_input_ids = input_ids[long_sentence_mask]
#         long_attention_mask = attention_mask[long_sentence_mask]
#         num_long_sentences = len(long_input_ids)
#         logging.info(
#             f"Cut {num_long_sentences} sentence with shape {long_input_ids.shape}"
#         )

#         if num_long_sentences > 0:
#             for i in range(len(long_input_ids)):
#                 long_sentence_buffer.append((long_input_ids[i], long_attention_mask[i]))
#             input_ids = input_ids[~long_sentence_mask]
#             attention_mask = attention_mask[~long_sentence_mask]
#             kv = [k[:, :, ~long_sentence_mask, :, :] for k in kv]
#             input_ids, attention_mask, kv = trim_inputs(input_ids, attention_mask, kv)

#     return input_ids, attention_mask, kv


# # def generator()
# import time


# def prepare(input_ids, attention_mask, kv):
#     # input_ids = inputs["input_ids"]
#     # attention_mask = inputs["attention_mask"]
#     input_ids = input_ids[:, -1:]
#     position_ids = attention_mask.long().cumsum(-1) - 1
#     position_ids.masked_fill_(attention_mask == 0, 1)
#     position_ids = position_ids[:, -1].unsqueeze(-1)
#     return input_ids, attention_mask, kv, position_ids


# # %%


# @torch.no_grad()
# def generate(model, data_generator, tokenizer):
#     model.eval()
#     get_funcs = [data_generator.send, data_generator.get_long_sentences]
#     mini_batches = [32, 16]
#     total_outputs = []
#     for stage in range(2):
#         mini_batch_size = mini_batches[stage]
#         get_func = get_funcs[stage]
#         input_ids, attention_mask = get_func(mini_batch_size, None)
#         out = forward_from_inputs(model, input_ids, attention_mask)
#         eos_token_id = tokenizer.eos_token_id

#         # long_sentence_buffer = []  # only save input_ids, attention_mask
#         long_sentence_buffer = data_generator.long_sentence_buffer

#         outputs = []
#         iter_end = False

#         # maintain a global index list of (batch size) to compute global index of each sentence
#         # global_idxs = list(range(mini_batch_size))
#         # last_global_idx = mini_batch_size - 1

#         while True:
#             start = time.time()
#             kv = list(out.past_key_values)
#             next_token_logits = out.logits[:, -1, :]
#             next_tokens = torch.argmax(next_token_logits, dim=-1)
#             input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
#             attention_mask = torch.cat(
#                 [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))],
#                 dim=-1,
#             )
#             this_token_eos = (next_tokens == eos_token_id).squeeze()
#             unfinished = ~this_token_eos
#             finished = input_ids[this_token_eos]
#             input_ids = input_ids[unfinished]
#             attention_mask = attention_mask[unfinished]
#             """
#             KV shape: [2, seq_len, batch_size, num_head, head_size] with mp head size
#             """
#             # kv = kv[:,:,unfinished,:,:]
#             # this function can be fused with `trim` to avoid creating twice when trimming
#             for i in range(len(kv)):
#                 kv[i] = kv[i][:, :, unfinished, :, :]

#             if stage == 0:
#                 input_ids, attention_mask, kv = cut_long_sentence(
#                     input_ids, attention_mask, kv, long_sentence_buffer, eos_token_id
#                 )

#             if len(finished) > 0:
#                 logging.info(f"finished: {len(finished)}")
#                 outputs.append(finished)

#                 # trim and cut long sentence
#                 input_ids, attention_mask, kv = trim_inputs(
#                     input_ids, attention_mask, kv
#                 )

#                 unfinished_batch_size, padding_length = input_ids.shape
#                 request_size = mini_batch_size - unfinished_batch_size
#                 if iter_end:
#                     logging.info("Finished part")
#                     if len(input_ids) <= 4:
#                         # outputs_idxs.extend(global_idxs) # in order generate without pop
#                         def generate_batch_to_end(input_ids, attention_mask, kv):
#                             # all is concated
#                             logging.info(
#                                 f"generate batch to end: {input_ids.shape}, {attention_mask.shape}, {kv[0].shape}"
#                             )
#                             # eos_token_id = 2
#                             while True:
#                                 out = forward_from_inputs(
#                                     model, *prepare(input_ids, attention_mask, kv)
#                                 )
#                                 kv = out.past_key_values
#                                 next_token_logits = out.logits[:, -1, :]
#                                 next_tokens = torch.argmax(next_token_logits, dim=-1)

#                                 last_tokens = input_ids[:, -1].squeeze()
#                                 last_token_eos = (last_tokens == eos_token_id).squeeze()
#                                 next_tokens[last_token_eos] = eos_token_id

#                                 input_ids = torch.cat(
#                                     [input_ids, next_tokens[:, None]], dim=-1
#                                 )
#                                 attention_mask = torch.cat(
#                                     [
#                                         attention_mask,
#                                         attention_mask.new_ones(
#                                             (attention_mask.shape[0], 1)
#                                         ),
#                                     ],
#                                     dim=-1,
#                                 )
#                                 this_token_eos = (next_tokens == eos_token_id).squeeze()
#                                 if torch.all(this_token_eos):
#                                     break
#                             return input_ids

#                         outputs.append(
#                             generate_batch_to_end(input_ids, attention_mask, kv)
#                         )
#                         logging.info("break since last 1/2 minibatch go to end")
#                         break
#                 else:
#                     try:
#                         _input_ids, _attention_mask = get_func(
#                             request_size, padding_length - 1
#                         )  # need one forward pass to align the kv cache

#                         # index operation
#                         # new_global_idxs = [last_global_idx + i + 1 for i in range(request_size)]
#                         # global_idxs.extend(new_global_idxs)
#                         # logging.info(f"new global idxs: {new_global_idxs} of length {len(new_global_idxs)}")
#                         # last_global_idx += request_size

#                         _position_ids = _attention_mask.long().cumsum(-1) - 1
#                         _position_ids.masked_fill_(_attention_mask == 0, 1)
#                         _out = model(
#                             input_ids=_input_ids,
#                             attention_mask=_attention_mask,
#                             position_ids=_position_ids,
#                             use_cache=True,
#                         )
#                         _next_token_logits = _out.logits[:, -1, :]
#                         _next_tokens = torch.argmax(_next_token_logits, dim=-1)
#                         _input_ids = torch.cat(
#                             [_input_ids, _next_tokens[:, None]], dim=-1
#                         )
#                         _attention_mask = torch.cat(
#                             [
#                                 _attention_mask,
#                                 _attention_mask.new_ones((_attention_mask.shape[0], 1)),
#                             ],
#                             dim=-1,
#                         )
#                         logging.info(
#                             f"New data forward finished with {_input_ids.shape}"
#                         )

#                         input_ids = torch.cat([input_ids, _input_ids], dim=0)
#                         attention_mask = torch.cat(
#                             [attention_mask, _attention_mask], dim=0
#                         )
#                         kv = concat_kv_on_batch(kv, _out.past_key_values)
#                     except StopIteration:
#                         logging.info("Data generator exhausted")
#                         iter_end = True

#             if iter_end and len(input_ids) == 0:
#                 logging.info("Break")
#                 break

#             out = forward_from_inputs(model, *prepare(input_ids, attention_mask, kv))

#         total_outputs.extend(outputs)
#     return total_outputs


# # %%

# data_generator = DataGenerator(tokenizer, prompts)
# outputs = generate(model, data_generator, tokenizer)

# # %%

# outputs[0]

# # %%

# # outputs[0]
# strs = []
# for out in outputs:
#     strs.extend(tokenizer.batch_decode(out, skip_special_tokens=True))
# strs

# # %%

# str_tuple = list(zip(strs, outputs_idx))
# str_tuple = sorted(str_tuple, key=lambda x: x[1])
# strs = [x[0] for x in str_tuple]
# strs
# # str_tuple

# # %%

# data_generator = DataGenerator(prompts, tokenizer)
# g_config = GenerationConfig(do_sample=False, max_length=4096)
# next(data_generator)

# # %%

# outputs = []
# for i in range(8):
#     start = time.time()
#     logging.info(f"Round {i+1}")
#     input_ids, attention_mask = data_generator.send((16, None))
#     seq = model.generate(
#         input_ids=input_ids,
#         attention_mask=attention_mask,
#         use_cache=True,
#         generation_config=g_config,
#     )
#     outputs.append(seq)
#     end = time.time()
#     logging.info(f"Time of batch 16: {end-start:.4f} seconds")

# # %%


# a = torch.ones((2, 2))
# m = torch.tensor([True, False])
# m & ~m

# # %%


# import torch
# import torch.nn.functional as F

# # 创建一个形状为 (batch_size, seq_len) 的张量
# tensor = torch.rand(5, 10)

# # 指定左 padding 的数量
# left_padding = 3

# # 对张量进行左 padding
# tensor = F.pad(tensor, (left_padding, 0, 0, 0))

# print(tensor.shape)  # 输出：torch.Size([5, 13])

# # %%

# tensor
