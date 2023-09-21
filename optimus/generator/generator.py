import torch
import logging
from collections import deque
import gc

import tree
from transformers.generation import LogitsWarper


def empty_cache():
    gc.collect()
    torch.cuda.empty_cache()


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


def greedy_decode(logits):  # -> [batch_size,]
    # logits come from model output, i.e. out.logits
    next_token_logits = logits[:, -1, :]
    next_tokens = torch.argmax(next_token_logits, dim=-1)
    return next_tokens


class top_p_decode:
    """
    [`LogitsWarper`] that performs top-p, i.e. restricting to top tokens summing to prob_cut_off <= prob_cut_off.

    Args:
        top_p (`float`):
            If set to < 1, only the smallest set of most probable tokens with probabilities that add up to `top_p` or
            higher are kept for generation.
        filter_value (`float`, *optional*, defaults to `-float("Inf")`):
            All filtered values will be set to this float value.
        min_tokens_to_keep (`int`, *optional*, defaults to 1):
            Minimum number of tokens that cannot be filtered.
    """

    def __init__(
        self,
        top_p: float = 0.9,
        filter_value: float = -float("Inf"),
        min_tokens_to_keep: int = 1,
    ):
        top_p = float(top_p)
        if top_p < 0 or top_p > 1.0:
            raise ValueError(f"`top_p` has to be a float > 0 and < 1, but is {top_p}")
        if not isinstance(min_tokens_to_keep, int) or (min_tokens_to_keep < 1):
            raise ValueError(
                f"`min_tokens_to_keep` has to be a positive integer, but is {min_tokens_to_keep}"
            )

        self.top_p = top_p
        self.filter_value = filter_value
        self.min_tokens_to_keep = min_tokens_to_keep

    def __call__(self, logits: torch.FloatTensor, top_p=None) -> torch.FloatTensor:
        scores = logits[:, -1, :]
        sorted_logits, sorted_indices = torch.sort(scores, descending=False)
        cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)

        # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
        top_p = top_p if top_p is not None else self.top_p
        sorted_indices_to_remove = cumulative_probs <= (1 - top_p)
        # Keep at least min_tokens_to_keep
        sorted_indices_to_remove[..., -self.min_tokens_to_keep :] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove
        )
        scores = scores.masked_fill(indices_to_remove, self.filter_value)

        probs = torch.nn.functional.softmax(scores, dim=-1)
        next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        return next_tokens


# %%


class Generator:
    def __init__(
        self,
        model,
        tokenizer,
        prompts,
        stages=None,
        max_length=2048,
        decode_fn=greedy_decode,
        forward_kwargs={
            "left_padding": True,
            "use_cache": True,
        },  # assume you are using a optimus hf model
    ):
        self.decode = decode_fn
        self.forward_kwargs = forward_kwargs
        # self.model = model.eval()
        model.eval()
        self.model = model
        self.prompts = prompts
        self.prompt_ptr = 0

        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.max_length = max_length

        """
        Stage is a list of tuple of (LENGTH, THRESHOULD, GENERATION_BATCH_SIZE)
        when batch of sequence reach THRESHOULD, this data manager
        would cut the sequence whose length > LENGTH,
        and `trim` the batch information, give new sequence into batch,
        then model will inference
        """
        _default_stages = [(400, 450, 128), (550, 600, 128), (700, 750, 64)]
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
        # next_token_logits = out.logits[:, -1, :]
        # next_tokens = torch.argmax(next_token_logits, dim=-1)
        next_tokens = self.decode(out.logits)
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        attention_mask = torch.cat(
            [
                attention_mask,
                attention_mask.new_ones((attention_mask.shape[0], 1)),
            ],
            dim=-1,
        )

        # concat to self states
        if self.input_ids is None:
            logging.info(f"REInit self states with shape {input_ids.shape}")
            self.input_ids = input_ids
            self.attention_mask = attention_mask
            self.kv = kv
            return

        self.input_ids = torch.cat([self.input_ids, input_ids], dim=0)
        self.attention_mask = torch.cat([self.attention_mask, attention_mask], dim=0)
        for i in range(len(self.kv)):
            self.kv[i] = torch.cat([self.kv[i], kv[i]], dim=2)

    @property
    def run_shape(self):
        if self.input_ids is None:
            return 0, 0
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
        if self.prompt_ptr >= len(self.prompts):
            raise StopIteration

        start = self.prompt_ptr
        end = self.prompt_ptr + size
        end = end if end <= len(self.prompts) else len(self.prompts)
        # if end > len(self.prompts):
        #     raise StopIteration
        selected = self.prompts[start:end]
        real_size = end - start
        idxs = range(start, end)
        self.prompt_ptr = end
        self.batch_global_idxs.extend(
            idxs
        )  # extend since next sentence is always been concated on the end of batch

        if padding_to is None:
            inputs = self.tokenizer(
                selected, return_tensors=None, padding="longest", truncation=False
            )
        else:
            inputs = self.tokenizer(
                selected,
                return_tensors=None,
                padding="max_length",
                truncation=True,
                max_length=padding_to,
            )
        inputs = tree.map_structure_up_to(
            {"input_ids": None, "attention_mask": None},
            lambda x: torch.tensor(x, dtype=torch.long, device="cuda").reshape(
                real_size, -1
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
                    # use_cache=True,
                    # left_padding=True,
                    **self.forward_kwargs,
                )
            else:
                out = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    # use_cache=True,
                    # left_padding=True,
                    **self.forward_kwargs,
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
        # next_token_logits = out.logits[:, -1, :]
        # next_tokens = torch.argmax(next_token_logits, dim=-1)
        next_tokens = self.decode(out.logits)
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
        batch_finished = num_finished == self.run_batch_size
        if batch_finished:
            logging.info(f"Batch finished with size {self.run_batch_size}")
            self.outputs.append(self.input_ids)
            self.output_idxs.append(self.batch_global_idxs)

            self.batch_global_idxs = []

            self.input_ids = None
            self.attention_mask = None
            self.kv = None
            empty_cache()
            return

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

            if len(long_sentence_batch_idxs) == self.run_batch_size:
                logging.info(f"Batch cutted with shape {self.run_shape}")

                for i in long_sentence_batch_idxs:
                    sequence = Sequence(
                        self.input_ids[i],
                        self.attention_mask[i],
                        self.batch_global_idxs[i],
                    )
                    self.stage_buffer.append(sequence)

                self.batch_global_idxs = []

                self.input_ids = None
                self.attention_mask = None
                self.kv = None
                empty_cache()
                return

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
        first_token = self.forward_pass_with_greedy_decode(
            input_ids, attention_mask, position_ids
        )  # use `self.kv`

        self.process_eos(first_token)

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
                        need_size, run_seq_len - 1 if run_seq_len > 1 else None
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
                # next_token_logits = out.logits[:, -1, :]
                # next_tokens = torch.argmax(next_token_logits, dim=-1)
                next_tokens = self.decode(out.logits)
                input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
                attention_mask = torch.cat(
                    [
                        attention_mask,
                        attention_mask.new_ones((attention_mask.shape[0], 1)),
                    ],
                    dim=-1,
                )

                if self.input_ids is None:
                    self.input_ids = input_ids
                    self.attention_mask = attention_mask
                    self.kv = kv

                else:
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

    def reach_max_length(self):
        return self.run_seq_len >= self.max_length

    def run_stage(self, i):
        empty_cache()
        if i != len(self.stages) - 1:  # not final stage
            # last_stage_capicity = len(self.stage_prefetch_buffer) + len(self.stage_buffer)
            last_stage_capicity = len(self.stage_buffer)
            last_stage_seq_len = self.stages[i][1]  # threshould
            num_picking_from_buffer = 0
            LONG_SENTECE_LENGTH, THRESHOLD, mini_batch_size = self.stages[i + 1]
            logging.warning(
                f"Running stage {i} with capacity {last_stage_capicity} of length {last_stage_seq_len}\n"
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
            # mini_batch_size = 32
            mini_batch_size = int(self.stages[i][2] / 2)
            while True:
                # 1. fill self state
                run_batch_size, run_seq_len = self.run_shape

                # 1.5: check whether reach max length
                if self.reach_max_length():
                    logging.warning(
                        f"Reach max length {self.max_length} of shape {self.run_shape}"
                    )
                    self.output_idxs.append(self.batch_global_idxs)
                    self.outputs.append(self.input_ids)

                    self.batch_global_idxs = []

                    self.input_ids = None
                    self.attention_mask = None
                    self.kv = None
                    empty_cache()

                    num_want = mini_batch_size
                    buffer_left = last_stage_capicity - num_picking_from_buffer
                    pick_size = min(buffer_left, num_want)
                    if num_picking_from_buffer >= last_stage_capicity:
                        logging.warning(
                            f"Reach max length {self.max_length}, and no more sentence"
                        )
                        # no more sentence in buffer
                        break
                    num_picking_from_buffer += pick_size
                    padding_length = None
                    self._get_buffer_sequences_and_fill_state(pick_size, padding_length)

                else:
                    num_want = mini_batch_size - run_batch_size
                    if num_want > 0:
                        buffer_left = last_stage_capicity - num_picking_from_buffer
                        pick_size = min(buffer_left, num_want)
                        if num_picking_from_buffer >= last_stage_capicity:
                            logging.warning(
                                f"Reach length {self.run_seq_len}, and no more sentence in buffer,\n"
                                f"Will go to epilogue"
                            )
                            # no more sentence in buffer
                            break
                        num_picking_from_buffer += pick_size
                        padding_length = run_seq_len - last_stage_seq_len - 1
                        padding_length = padding_length if padding_length > 0 else None
                        self._get_buffer_sequences_and_fill_state(
                            pick_size, padding_length
                        )

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

            if self.input_ids is None:
                return

            logging.warning(
                f"Runnning epilogue with seqlen {self.run_seq_len}, batch size {self.run_batch_size}"
            )
            while self.run_batch_size > 0 and self.run_seq_len < self.max_length:
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

            if self.input_ids is not None:
                self.outputs.append(self.input_ids)
                self.output_idxs.append(self.batch_global_idxs)

    def get_output(self):
        return self.outputs, self.output_idxs

    def flush(self):
        self.input_ids = None
        self.attention_mask = None
        self.kv = None
        empty_cache()

    def run(self):
        self.run_init_stage()
        for i in range(len(self.stages)):
            self.run_stage(i)

        output, index = self.get_output()

        outs = []
        for o, i in zip(output, index):
            for t, p in zip(o, i):
                text = self.tokenizer.decode(t)
                idx = p
                outs.append((text, idx))

        outs = sorted(outs, key=lambda x: x[1])
        self.flush()
        return [o[0] for o in outs]  # str list
