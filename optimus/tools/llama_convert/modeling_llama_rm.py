from transformers import PreTrainedModel, LlamaConfig, LlamaModel
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm
import torch.nn as nn
import torch
from typing import Optional

class LlamaRewardModel(PreTrainedModel):
    config_class = LlamaConfig
    _no_split_modules = ["LlamaDecoderLayer", "value_head"]

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.value_head = nn.Linear(config.hidden_size, 1, bias=False)
        print("LlamaRewardModel init")

    def forward(
        self, input_ids: torch.LongTensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        outputs = self.model(
            input_ids, attention_mask=attention_mask, output_hidden_states=True
        )
        last_hidden_states = outputs.hidden_states[-1]
        if attention_mask is None:
            last_hidden_states = last_hidden_states[:, -1]
        else:
            last_index = attention_mask.cumsum(dim=1).argmax(dim=1)
            last_hidden_states = last_hidden_states.to(last_index.device)
            last_hidden_states = last_hidden_states.gather(
                1, last_index.view(-1, 1, 1).expand(-1, 1, last_hidden_states.size(-1))
            ).squeeze(1)
        values = self.value_head(last_hidden_states).squeeze(-1)  # (bs,)

        return values
