# %%

import time

# from transformers.models.gpt_neo.configuration_gpt_neo import OrderedDict

start = time.time()

import torch
from ipytorch import logging
from tqdm.auto import tqdm
from collections import OrderedDict
from transformers import AutoTokenizer
from modeling_baichuan import BaichuanConfig
from optimus.hf.baichuan import BaichuanConfig as BaichuanConfigNeox
from functools import partial
from pathlib import Path
import sys
import json

# %%


from absl import app, flags
from absl.app import _run_init, parse_flags_with_usage
from transformers.modeling_utils import shard_checkpoint
import shutil

FLAGS = flags.FLAGS

flags.DEFINE_string("model_path", None, "MP model path", short_name="m")

flags.DEFINE_string("output_path", None, "HF model output path", short_name="o")

flags.DEFINE_string("hf_path", None, "dir to hf ckpt to get config", short_name="h")

flags.DEFINE_string(
    "model_type",
    "causal",
    "model type (`causal`, `backbone` or `reward`)",
    short_name="t",
)
# flags.DEFINE_bool("vhead_with_bias", False, "Value Head With BI")


args = _run_init(sys.argv, parse_flags_with_usage)

# %%

CKPT_PATH = Path(FLAGS.model_path)
SAVE_DIR = Path(FLAGS.output_path)
SAVE_DIR.mkdir(exist_ok=True, parents=True)
HF_PATH = Path(FLAGS.hf_path)

_types = ["causal", "backbone", "reward"]
MODEL_TYPE = FLAGS.model_type
assert MODEL_TYPE in _types, f"model_type must be one of {_types}"

logging.info(FLAGS)

# %%


Hf_LLAMA = "model"
Neox_LLAMA = "llama"

Hf_EMBED_IN_KEY = "embed_tokens.weight"
Noex_EMBED_IN_KEY = "embed_in.word_embeddings.weight"

Hf_FINAL_NORM_KEY = "norm.weight"
Neox_FINAL_NORM_KEY = "final_layer_norm.scale"


# %%

hf_config = BaichuanConfig.from_pretrained(HF_PATH)

num_heads = hf_config.num_attention_heads
hidden_size = hf_config.hidden_size
dims_per_head = hidden_size // num_heads
mp_partitions = 4

# %%

WEIGHTS_INDEX_NAME = "pytorch_model.bin.index.json"
states = [OrderedDict() for _ in range(mp_partitions)]
concat_weight = OrderedDict()
for i in tqdm(range(mp_partitions)):
    ckpt_path = CKPT_PATH / f"part_{i}"
    index_file = ckpt_path / WEIGHTS_INDEX_NAME
    loader = partial(torch.load, map_location="cpu")
    if not index_file.exists():
        # must have a pytorch_model.bin
        ckpt_file = ckpt_path / "pytorch_model.bin"
        assert ckpt_file.exists(), f"{ckpt_file} does not exist"
        state_dict = loader(ckpt_file)
        states[i].update(state_dict)

    else:
        with open(index_file, "r", encoding="utf-8") as f:
            index = json.load(f)
        shard_files = list(set(index["weight_map"].values()))
        for shard_file in shard_files:
            state_dict = loader(ckpt_path / str(shard_file))
            states[i].update(state_dict)

# %%

"""
LLAMA backbone convert
"""

concat_weight[f"{Hf_LLAMA}.{Hf_EMBED_IN_KEY}"] = torch.cat(
    [t[f"{Neox_LLAMA}.{Noex_EMBED_IN_KEY}"] for t in states], dim=0
)
concat_weight[f"{Hf_LLAMA}.{Hf_FINAL_NORM_KEY}"] = (
    sum([t[f"{Neox_LLAMA}.{Neox_FINAL_NORM_KEY}"] for t in states]) / mp_partitions
)

for i in tqdm(range(hf_config.num_hidden_layers)):
    hf_layer_prefix = f"{Hf_LLAMA}.layers.{i}"
    neox_layer_prefix = f"{Neox_LLAMA}.layers.{i}"

    sharded_qkv = torch.cat(
        [
            state[f"{neox_layer_prefix}.attention.query_key_value.weight"]
            for state in states
        ],
        dim=0,
    )
    sharded_qkv = sharded_qkv.view(num_heads, 3, dims_per_head, hidden_size)
    q, k, v = sharded_qkv.chunk(3, dim=1)
    q = q.reshape(num_heads * dims_per_head, hidden_size)
    k = k.reshape(num_heads * dims_per_head, hidden_size)
    v = v.reshape(num_heads * dims_per_head, hidden_size)
    qkv = torch.cat([q, k, v], dim=0)
    concat_weight[f"{hf_layer_prefix}.self_attn.W_pack.weight"] = qkv
    # concat_weight[f"{hf_layer_prefix}.self_attn.q_proj.weight"] = q.reshape(
    #     num_heads * dims_per_head, hidden_size
    # )
    # concat_weight[f"{hf_layer_prefix}.self_attn.k_proj.weight"] = k.reshape(
    #     num_heads * dims_per_head, hidden_size
    # )
    # concat_weight[f"{hf_layer_prefix}.self_attn.v_proj.weight"] = v.reshape(
    #     num_heads * dims_per_head, hidden_size
    # )

    concat_weight[f"{hf_layer_prefix}.self_attn.o_proj.weight"] = torch.cat(
        [t[f"{neox_layer_prefix}.attention.dense.weight"] for t in states], dim=1
    )
    concat_weight[f"{hf_layer_prefix}.self_attn.rotary_emb.inv_freq"] = states[0][
        f"{neox_layer_prefix}.attention.rotary_emb.inv_freq"
    ]

    # average layernorm stats over mp ranks
    concat_weight[f"{hf_layer_prefix}.input_layernorm.weight"] = (
        sum([t[f"{neox_layer_prefix}.input_layernorm.scale"] for t in states])
    ) / mp_partitions
    concat_weight[f"{hf_layer_prefix}.post_attention_layernorm.weight"] = (
        sum([t[f"{neox_layer_prefix}.post_attention_layernorm.scale"] for t in states])
    ) / mp_partitions

    # mlp params
    concat_weight[f"{hf_layer_prefix}.mlp.gate_proj.weight"] = torch.cat(
        [t[f"{neox_layer_prefix}.mlp.w1.weight"] for t in states], dim=0
    )
    concat_weight[f"{hf_layer_prefix}.mlp.up_proj.weight"] = torch.cat(
        [t[f"{neox_layer_prefix}.mlp.w3.weight"] for t in states], dim=0
    )
    concat_weight[f"{hf_layer_prefix}.mlp.down_proj.weight"] = torch.cat(
        [t[f"{neox_layer_prefix}.mlp.w2.weight"] for t in states], dim=1
    )

# %%

if MODEL_TYPE == "reward":
    # FOR VALUE HEAD, suppose value head do not split, init by each mp rank
    concat_weight[f"value_head.weight"] = states[0]["value_head.weight"]
    if "value_head.bias" in states[0]:
        concat_weight[f"value_head.bias"] = states[0]["value_head.bias"]

elif MODEL_TYPE == "causal":
    concat_weight[f"lm_head.weight"] = torch.cat(
        [t["embed_out.final_linear.weight"] for t in states], dim=0
    )

else:  # backbone
    pass

# %%


neox_config = BaichuanConfigNeox.from_pretrained(CKPT_PATH / "part_0")
# tokenizer = LlamaTokenizer.from_pretrained(CKPT_PATH / "part_0")
tokenizer = AutoTokenizer.from_pretrained(HF_PATH, trust_remote_code=True, use_fast=False)

def convert_config(neox_config, rm=True):
    hf_config = BaichuanConfig(
        vocab_size=neox_config.vocab_size,
        hidden_size=neox_config.hidden_size,
        intermediate_size=neox_config.intermediate_size,
        num_hidden_layers=neox_config.num_hidden_layers,
        num_attention_heads=neox_config.num_attention_heads,
        hidden_act="silu",
        max_position_embeddings=2048,
        initializer_range=0.02,
        rms_norm_eps=neox_config.rms_norm_epsilon,
        use_cache=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        torch_dtype=neox_config.torch_dtype,
        # rope_scaling={"type": "linear", "factor": neox_config.rotary_pct},
    )
    if rm:
        hf_config.auto_map = {
            "AutoModelForSequenceClassification": "modeling_llama_rm.LlamaRewardModel"
        }
    if MODEL_TYPE == "causal":
        hf_config.architectures = ["LlamaForCausalLM"]
    return hf_config


hf_config = convert_config(neox_config, rm=MODEL_TYPE == "reward")
shards, index = shard_checkpoint(concat_weight, "5GB")
with open(SAVE_DIR / "pytorch_model.bin.index.json", "w", encoding="utf-8") as f:
    content = json.dumps(index, indent=2, sort_keys=True) + "\n"
    f.write(content)

for shard_file, shard in tqdm(shards.items()):
    torch.save(shard, SAVE_DIR / str(shard_file))


tokenizer.save_pretrained(SAVE_DIR)
hf_config.save_pretrained(SAVE_DIR)
shutil.copy(Path("modeling_llama_rm.py"), SAVE_DIR / "modeling_llama_rm.py")

# %%

used = time.time() - start
print(f"Convert used {used:.2f}s")
