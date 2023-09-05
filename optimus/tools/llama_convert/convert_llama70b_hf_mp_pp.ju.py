# %%

import torch
import json
from transformers import LlamaForCausalLM, LlamaConfig
from megatron.modeling_llama_neox import LlamaConfig as NeoxConfig
from transformers.modeling_utils import shard_checkpoint
from transformers import AutoTokenizer
from pathlib import Path
from collections import OrderedDict

# %%

import sys
from absl import app, flags
from absl.app import _run_init, parse_flags_with_usage

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "model_path", None, "hf ckpt path", short_name="m"
)

flags.DEFINE_string(
    "output_path", None, "mp dst dir", short_name="o"
)
flags.DEFINE_integer(
    "model_parallel_size", 8, "model parallel size", short_name="p"
)

args = _run_init(sys.argv, parse_flags_with_usage)

# %%

HF_DIR = Path(FLAGS.model_path)
SAVE_DIR = Path(FLAGS.output_path)
SAVE_DIR.mkdir(exist_ok=True)
MP_SIZE = int(FLAGS.model_parallel_size) # 8

# %%

hf_config = LlamaConfig.from_pretrained(HF_DIR)
tokenizer = AutoTokenizer.from_pretrained(HF_DIR)
hf_model = LlamaForCausalLM.from_pretrained(
    HF_DIR, device_map="cpu", torch_dtype=torch.float16
)

# %%

def getattr_if(config, options, value=True):
    for option in options:
        if hasattr(config, option):
            if value:
                return getattr(config, option)
            return {option: getattr(config, option)}
    return None


def convert_config(hf_config: LlamaConfig, isGQA: bool = False):
    neox_config = NeoxConfig(
        vocab_size=hf_config.vocab_size,
        hidden_size=hf_config.hidden_size,
        num_hidden_layers=hf_config.num_hidden_layers,
        num_attention_heads=hf_config.num_attention_heads,
        intermediate_size=hf_config.intermediate_size,
        hidden_act=hf_config.hidden_act,
        rotary_pct=1,
        rotary_emb_base=10000,
        max_position_embeddings=hf_config.max_position_embeddings,
        initializer_range=hf_config.initializer_range,
        rms_norm_epsilon=hf_config.rms_norm_eps,
        torch_dtype=hf_config.torch_dtype,
        use_cache=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        use_parallel_residual=False,
    )
    neox_config.llama_mlp_multiple_of = 256
    assert (
        neox_config.intermediate_size % neox_config.llama_mlp_multiple_of == 0
    ), f"{neox_config.intermediate_size} % {neox_config.llama_mlp_multiple_of}"
    neox_config.init_method = "small_init"
    neox_config.hidden_dropout = 0
    neox_config.output_layer_init_method = "wang_init"
    neox_config.pos_emb = "rotary"
    neox_config.norm = "rmsnorm"
    neox_config.gpt_j_residual = False
    neox_config.gpt_j_tied = False
    neox_config.apply_query_key_layer_scaling = False
    neox_config.attention_softmax_in_fp32 = False
    neox_config.scaled_masked_softmax_fusion = True
    neox_config.scaled_upper_triang_masked_softmax_fusion = False
    neox_config.bias_gelu_fusion = False
    neox_config.attention_dropout = 0
    neox_config.output_layer_parallelism = "column"
    neox_config.eod_mask_loss = False
    neox_config.bias_dropout_fusion = False
    neox_config.attention_config = [[["flash"], "all"]]
    neox_config.mlp_type = "llama"
    neox_config.use_bias_in_attn_linear = False
    neox_config.lora = False

    neox_config.isGQA = isGQA
    num_kv_heads = getattr_if(hf_config, ["num_kv_heads", "num_key_value_heads"])
    if num_kv_heads:
        neox_config.num_kv_heads = num_kv_heads
    return neox_config

# %%

mp_state_dict_list = [OrderedDict() for _ in range(MP_SIZE)]

# %%

def _select(keys, dct, map_fn=None, values=False):
    if map_fn is None:
        map_fn = lambda x: x
    if not values:
        return {k: map_fn(v) for k, v in dct.items() if k in keys}
    else:
        lst = [map_fn(v) for k, v in dct.items() if k in keys]
        if len(lst) == 1:
            return lst[0]
        else:
            return lst


def select_keyword(keyword, dct, map_fn=None, values=False):
    keys = [k for k in dct.keys() if keyword in k]
    return _select(keys, dct, map_fn, values)


# %%


def NoeXTransformerLayerPrefix(i, llama_prefix="llama.layers"):
    qkv = f"{llama_prefix}.{i}.attention.query_key_value.weight"
    dense = f"{llama_prefix}.{i}.attention.dense.weight"
    inv_frq = f"{llama_prefix}.{i}.attention.rotary_emb.inv_freq"

    i_layer_norm = f"{llama_prefix}.{i}.input_layernorm.scale"
    p_layer_norm = f"{llama_prefix}.{i}.post_attention_layernorm.scale"

    w1 = f"{llama_prefix}.{i}.mlp.w1.weight"
    w2 = f"{llama_prefix}.{i}.mlp.w2.weight"
    w3 = f"{llama_prefix}.{i}.mlp.w3.weight"
    return qkv, dense, inv_frq, i_layer_norm, p_layer_norm, w1, w2, w3


def split_transformer_layer(
    layer_params, layer_num, mp_state_dict_list, neox_config, num_partitions=MP_SIZE
):
    # layer_params = select_keyword(layer_prefix, state_dict) outer
    """
    All GEMM of shape (M,N,K) follows the following in HFLlama/Megatron(NeoX):
    1. `input` with hidden dim: hidden_dim means K on tensor dim -1
    2. `weight` maps hidden_dim -> output_size has shape (output_size, K) which means (N, K)
    3. `ColumnParallelLinear` split N dim, that means split dim 0 of weight
    4. `RowParallelLinear` split K dim, that means split dim 1 of weight
    5. `RowParallelLinear` would split K dim, we need to split GEMM-k-axis of (M,K), that
        means the -1 dim of tensor input, to calculate (M, N, k_1),...,(M, N, k_n) GEMM then
        use `all reduce` to get the result.
        Note that split K dim of the weight means split `dim 1` of the weight
    """

    # QKV & O
    num_q_heads_per_partition = neox_config.num_attention_heads // num_partitions
    num_kv_heads_per_partition = neox_config.num_kv_heads // num_partitions
    head_dim = neox_config.hidden_size // neox_config.num_attention_heads
    q_dim_per_partition = head_dim * num_q_heads_per_partition
    kv_dim_per_partition = head_dim * num_kv_heads_per_partition
    o_dim_per_partition = neox_config.hidden_size // num_partitions

    q = select_keyword("q_proj", layer_params, values=True)
    k = select_keyword("k_proj", layer_params, values=True)
    v = select_keyword("v_proj", layer_params, values=True)
    o = select_keyword("o_proj", layer_params, values=True)

    q_split = torch.split(q, q_dim_per_partition, dim=0)
    k_split = torch.split(k, kv_dim_per_partition, dim=0)
    v_split = torch.split(v, kv_dim_per_partition, dim=0)
    o_split = torch.split(o, o_dim_per_partition, dim=1)  # along k dim

    # Layer Norm
    input_layer_norm = select_keyword("input_layernorm", layer_params, values=True)
    post_attention_layernorm = select_keyword(
        "post_attention_layernorm", layer_params, values=True
    )

    # Llama MLP
    # w1 -> gate, w2 -> down, w3 -> up
    w1 = select_keyword("gate_proj", layer_params, values=True)
    w2 = select_keyword("down_proj", layer_params, values=True)
    w3 = select_keyword("up_proj", layer_params, values=True)

    w_dim_per_partition = neox_config.intermediate_size // num_partitions
    w1_split = torch.split(w1, w_dim_per_partition, dim=0)
    w3_split = torch.split(w3, w_dim_per_partition, dim=0)
    w2_split = torch.split(w2, w_dim_per_partition, dim=1)  # along k dim

    # inv_freq
    inv_freq = select_keyword("inv_freq", layer_params, values=True)

    (
        qkv_name,
        dense_name,
        inv_frq_name,
        i_layer_norm_name,
        p_layer_norm_name,
        w1_name,
        w2_name,
        w3_name,
    ) = NoeXTransformerLayerPrefix(layer_num)

    for rank in range(num_partitions):
        rank_dict = mp_state_dict_list[rank]
        qkv = torch.concat([q_split[rank], k_split[rank], v_split[rank]], dim=0)
        rank_dict[qkv_name] = qkv.clone()
        rank_dict[dense_name] = o_split[rank].clone()
        rank_dict[w1_name] = w1_split[rank].clone()
        rank_dict[w2_name] = w2_split[rank].clone()
        rank_dict[w3_name] = w3_split[rank].clone()

        rank_dict[inv_frq_name] = inv_freq
        rank_dict[i_layer_norm_name] = input_layer_norm
        rank_dict[p_layer_norm_name] = post_attention_layernorm


def split_embedding_layer(hf_state_dict, mp_stata_dict_list, num_partitions=MP_SIZE):
    embedding = select_keyword("embed", hf_state_dict, values=True)
    embedding_name = "llama.embed_in.word_embeddings.weight"
    embedding_split = torch.split(
        embedding, embedding.shape[0] // num_partitions, dim=0
    )

    for rank in range(num_partitions):
        mp_stata_dict_list[rank][embedding_name] = embedding_split[rank].clone()


def split_final_layernorm(hf_state_dict, mp_state_dict_list, num_partitions=MP_SIZE):
    layernorm = select_keyword("model.norm", hf_state_dict, values=True)
    layernorm_name = "llama.final_layer_norm.scale"
    for rank in range(num_partitions):
        mp_state_dict_list[rank][layernorm_name] = layernorm.clone()


def split_causal_layer(hf_state_dict, mp_stata_dict_list, num_partitions=MP_SIZE):
    causal = select_keyword("lm_head", hf_state_dict, values=True)
    causal_name = "embed_out.final_linear.weight"
    causal_split = torch.split(causal, causal.shape[0] // num_partitions, dim=0)
    for rank in range(num_partitions):
        mp_stata_dict_list[rank][causal_name] = causal_split[rank].clone()


# %%

neox_config = convert_config(hf_config, isGQA=True)
hf_state_dict = hf_model.state_dict()
neox_config

# %%

from tqdm.auto import tqdm

"""
Follow Llama Structure
"""
split_embedding_layer(hf_state_dict, mp_state_dict_list)
for layer_i in tqdm(range(neox_config.num_hidden_layers)):
    layer_params = select_keyword(f"layers.{layer_i}.", hf_state_dict)
    split_transformer_layer(layer_params, layer_i, mp_state_dict_list, neox_config)
split_final_layernorm(hf_state_dict, mp_state_dict_list)
split_causal_layer(hf_state_dict, mp_state_dict_list)

# %%


save_dir = SAVE_DIR
for rank in tqdm(range(MP_SIZE)):
    rank_dir = save_dir / f"part_{rank}"
    rank_dir.mkdir(exist_ok=True)
    sd = mp_state_dict_list[rank]
    shards, index = shard_checkpoint(sd, "5GB")
    with open(rank_dir / "pytorch_model.bin.index.json", "w", encoding="utf-8") as f:
        content = json.dumps(index, indent=2, sort_keys=True) + "\n"
        f.write(content)
    for shard_file, shard in shards.items():
        torch.save(shard, rank_dir / str(shard_file))
    neox_config.save_pretrained(rank_dir)
    tokenizer.save_pretrained(rank_dir)
