# Copyright 2024 PRIME team and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Implement a multiprocess PPOCritic
"""

import itertools

import torch
import torch.distributed
import torch.nn.functional as F
from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
from torch import nn, optim
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches
from verl.utils.ulysses import gather_outpus_and_unpad, ulysses_pad_and_slice_inputs
from verl.utils import hf_tokenizer

from verl.workers.prime_core_algos import compute_ce_dpo_loss_rm, compute_detach_dpo_loss_rm


class DataParallelProcessRewardModel:
    def __init__(self, config, reward_module: nn.Module, ref_module: nn.Module, reward_optimizer: optim.Optimizer):
        self.config = config
        self.reward_module = reward_module
        self.ref_module = ref_module
        self.reward_optimizer = reward_optimizer
        self.use_remove_padding = self.config.model.get("use_remove_padding", False)
        print(f"Reward model use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.model.get("use_fused_kernels", False)
        print(f"Reward model use_fused_kernels={self.use_fused_kernels}")
        self.freeze = self.config.model.get("freeze", False)
        print(f"Freeze process reward model={self.freeze}")

        self.ulysses_sequence_parallel_size = self.config.get("ulysses_sequence_parallel_size", 1)

        self.tokenizer = hf_tokenizer(self.config.model.path, trust_remote_code=self.config.model.get("trust_remote_code", False))

        
        if self.use_fused_kernels:
            from verl.utils.experimental.torch_functional import FusedLinearForPPO

            self.fused_linear_for_ppo = FusedLinearForPPO()

    def _forward_micro_batch(self, micro_batch, prompt_length):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]

            num_actions = micro_batch["responses"].size(-1)
            max_positions = micro_batch["attention_mask"][:, prompt_length:].sum(-1)

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices).transpose(0, 1)

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.ulysses_sequence_parallel_size > 1:
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(input_ids_rmpad, position_ids_rmpad, sp_size=self.ulysses_sequence_parallel_size)
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(input_ids_rmpad_rolled, None, self.ulysses_sequence_parallel_size)

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)
                output = self.reward_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    use_cache=False,
                )

                if self.use_fused_kernels:
                    hidden_states = output.last_hidden_state
                    vocab_weights = self.reward_module.lm_head.weight

                    rm_log_labels, _ = self.fused_linear_for_ppo(
                        hidden_states=hidden_states.squeeze(0),
                        vocab_weights=vocab_weights,
                        input_ids=input_ids_rmpad_rolled,
                    )
                    rm_log_labels = rm_log_labels.to(torch.float32)

                else:
                    rm_output_logits = output.logits.squeeze(0)
                    rm_log_labels = verl_F.logprobs_from_logits(
                        logits=rm_output_logits,
                        labels=input_ids_rmpad_rolled,
                    )

                if self.ulysses_sequence_parallel_size > 1:
                    rm_log_labels = gather_outpus_and_unpad(rm_log_labels, gather_dim=0, unpad_dim=0, padding_size=pad_size)
                rm_log_labels = pad_input(hidden_states=rm_log_labels.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen).squeeze(-1)[:, -num_actions - 1 : -1]

            else:
                output = self.reward_module(
                    input_ids=micro_batch["input_ids"],
                    attention_mask=micro_batch["attention_mask"],
                    position_ids=micro_batch["position_ids"],
                    use_cache=False,
                )

                if self.use_fused_kernels:
                    hidden_states = output.last_hidden_state
                    vocab_weights = self.reward_module.lm_head.weight

                    rm_log_labels, _ = self.fused_linear_for_ppo.forward(
                        hidden_states=hidden_states[:, :-1, :],
                        vocab_weights=vocab_weights,
                        input_ids=micro_batch["input_ids"][:, 1:],
                    )
                    rm_log_labels = rm_log_labels.to(torch.float32)

                else:
                    rm_output_logits = output.logits
                    rm_log_prob = torch.nn.functional.log_softmax(rm_output_logits[:, :-1, :], dim=-1)  # (batch_size, seq_length, vocab_size)
                    rm_log_labels = rm_log_prob.gather(dim=-1, index=micro_batch["input_ids"][:, 1:].unsqueeze(-1)).squeeze(-1)  # (batch, seq_length)

            if self.freeze: # if freeze do not compute ref_prob
                rm_log_labels = rm_log_labels.detach()
                q = rm_log_labels[:, -num_actions:]
                return q

            if self.ref_module is not None:
                # do not have to pad again
                with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    if self.ulysses_sequence_parallel_size > 1 and self.use_remove_padding:
                        ref_output = self.ref_module(
                            input_ids=input_ids_rmpad,
                            attention_mask=None,
                            position_ids=position_ids_rmpad,
                            use_cache=False,
                            )

                        if self.use_fused_kernels:
                            hidden_states = ref_output.last_hidden_state
                            vocab_weights = self.ref_module.lm_head.weight

                            ref_log_labels, _ = self.fused_linear_for_ppo(
                                hidden_states=hidden_states.squeeze(0),
                                vocab_weights=vocab_weights,
                                input_ids=input_ids_rmpad_rolled,
                            )
                            ref_log_labels = ref_log_labels.to(torch.float32)

                        else:
                            logits = ref_output.logits.squeeze(0)
                            ref_log_labels = verl_F.logprobs_from_logits(logits=ref_output_logits, labels=input_ids_rmpad_rolled)

                        ref_log_labels = gather_outpus_and_unpad(ref_log_labels, gather_dim=0, unpad_dim=0, padding_size=pad_size)
                        ref_log_labels = pad_input(hidden_states=ref_log_labels.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen).squeeze(-1)[:, -num_actions - 1 : -1]
                    else:
                        ref_output = self.ref_module(
                            input_ids=micro_batch["input_ids"],
                            attention_mask=micro_batch["attention_mask"],
                            position_ids=micro_batch["position_ids"],
                            use_cache=False,
                        )

                        if self.use_fused_kernels:
                            hidden_states = ref_output.last_hidden_state
                            vocab_weights = self.ref_module.lm_head.weight

                            ref_log_labels, _ = self.fused_linear_for_ppo.forward(
                                hidden_states=hidden_states[:, :-1, :],
                                vocab_weights=vocab_weights,
                                input_ids=micro_batch["input_ids"][:, 1:],
                            )
                            ref_log_labels = ref_log_labels.to(torch.float32)

                        else:
                            ref_output_logits = ref_output.logits
                            ref_log_prob = torch.nn.functional.log_softmax(ref_output_logits[:, :-1, :], dim=-1)  # (batch_size, seq_length, vocab_size)
                            ref_log_labels = ref_log_prob.gather(dim=-1, index=micro_batch["input_ids"][:, 1:].unsqueeze(-1)).squeeze(-1)  # (batch, seq_length)

            else:
                ref_log_labels = micro_batch["old_log_probs"]

            ref_log_labels.to(rm_log_labels.dtype)
            q = rm_log_labels[:, -num_actions:] - ref_log_labels[:, -num_actions:]  # this is actually diff of q

            # trim unnecessary logprobs here
            for i in range(micro_batch["input_ids"].shape[0]):
                q[i, max_positions[i] :] = 0

        return q

    def _optimizer_step(self):
        assert self.config.model.optim.grad_clip is not None

        if isinstance(self.reward_module, FSDP):
            grad_norm = self.reward_module.clip_grad_norm_(self.config.model.optim.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.reward_module.parameters(), max_norm=self.config.model.optim.grad_clip)
        self.reward_optimizer.step()
        return grad_norm

    def prime_norm(self, token_level_scores):
        if self.config.prime_norm == "batch_norm":
            reverse_cumsum = torch.cumsum(token_level_scores.flip(dims=[1]), dim=-1).flip(dims=[1])
            token_level_scores = token_level_scores / (reverse_cumsum.abs().max() + 1e-6)
        return token_level_scores

    def make_step_rewards(self, logits, token_masks):
        """
        Extracts step rewards from PRM model outputs efficiently.
        """
        device = logits.device
        batch_size = logits.size(0)
        max_steps = self.config.model.get("max_steps", 100)

        probabilities = F.softmax(logits, dim=-1)
        positive_probs = probabilities[:, :, 1]  # Shape: (batch_size, seq_len)

        rewards_tensor = torch.full(
            (batch_size, max_steps), 
            -99.0, 
            device=device, 
            dtype=positive_probs.dtype
        )

        col_indices = torch.cumsum(token_masks, dim=1) - 1

        placement_mask = (token_masks) & (col_indices < max_steps)

        row_indices = torch.arange(batch_size, device=device).unsqueeze(1).expand_as(placement_mask)

        rewards_tensor[row_indices[placement_mask], col_indices[placement_mask]] = positive_probs[placement_mask]

        return rewards_tensor
    
    
    def compute_prm_step_rewards(self, micro_batch, prompt_length):
        """
        Computes step rewards for a micro_batch.
        """
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            tokenizer = self.tokenizer

            device = self.reward_module.device if hasattr(self.reward_module, 'device') else 'cuda'
        
            # --- Batch preparation ---
            batch_input_ids = micro_batch["input_ids"]
            prompt_ids = batch_input_ids[:, :prompt_length]
            response_ids = batch_input_ids[:, prompt_length:]
            prompt_texts = tokenizer.batch_decode(prompt_ids, skip_special_tokens=True)
            response_texts = tokenizer.batch_decode(response_ids, skip_special_tokens=True)
            batch_of_strings = []
            system_msg = "Please reason step by step, and put your final answer within \\boxed{{}}."
            for prompt_text, response_text in zip(prompt_texts, response_texts):
                query = prompt_text.replace(system_msg, "").strip()
                response_steps = [s.strip() for s in response_text.strip().split('\n\n') if s.strip()]
                if not response_steps:
                    response_steps = [response_text.strip()]
                messages = [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": query},
                    {"role": "assistant", "content": "<extra_0>".join(response_steps) + "<extra_0>"},
                ]
                conversation_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
                batch_of_strings.append(conversation_str)
        
            # --- Batch Encoding & Model Call ---
            tokenizer.padding_side = 'left'
            inputs = tokenizer(
                batch_of_strings,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.model.get("max_length", 4096),
            ).to(device)
        
            with torch.no_grad():
                # CORRECTED: Passed both 'input_ids' and 'attention_mask' from the 'inputs' object.
                outputs = self.reward_module(
                    input_ids=inputs['input_ids']
                )

            step_sep_id = tokenizer.encode("<extra_0>")[0]
            token_masks = (inputs['input_ids'] == step_sep_id)
        
            rewards_tensor = self.make_step_rewards(outputs[0], token_masks)
                    
            return rewards_tensor


    def compute_rm_score(self, data: DataProto):
        # self.reward_module.eval()
        # device = data.batch["input_ids"].device
        # batch_size = data.batch["input_ids"].size(0)
        # max_steps = self.config.model.get("max_steps", 100)
        # q = torch.full((batch_size, max_steps), -99.0, device=device)
        # q[:, :10] = 0.0
        # mask = (q != -99.0)
        # return (
        #     q.detach(),
        #     {
        #         "reward_model/raw_reward": (q * mask).sum(dim=-1).mean().item(),
        #     },
        # )
        return self.compute_prm_verify_score(data)
        #return self.compute_original_rm_score(data)

    def compute_prm_verify_score(self, data: DataProto):
        """Compute rewards using PRM model in verification mode."""
        self.reward_module.eval()
        micro_batch_size = data.meta_info["micro_batch_size"]
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids", "seq_reward"]
        batch = data.select(batch_keys=select_keys).batch
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        prompt_length = data.batch["input_ids"].shape[-1] - data.batch["responses"].shape[-1]

        if use_dynamic_bsz:
            # split using dynamic bsz
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)

        q_lst = []
        for micro_batch in micro_batches:
            with torch.no_grad():
                # Use PRM model to compute step rewards
                step_rewards = self.compute_prm_step_rewards(micro_batch, prompt_length)

            q_lst.append(step_rewards)

        q = torch.concat(q_lst, dim=0)
            
        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == q.size(0), f"{len(indices)} vs. {q.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            q = q[revert_indices]

        mask = (q != -99.0)

        return (
            q.detach(),
            {
                "reward_model/raw_reward": (q * mask).sum(dim=-1).mean().item(),
            },
        )

    def compute_original_rm_score(self, data: DataProto):
        """Original compute_rm_score method."""
        self.reward_module.eval()
        self.ref_module.eval()
        micro_batch_size = data.meta_info["micro_batch_size"]
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids", "seq_reward"]
        batch = data.select(batch_keys=select_keys).batch
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        prompt_length = data.batch["input_ids"].shape[-1] - data.batch["responses"].shape[-1]

        if use_dynamic_bsz:
            # split using dynamic bsz
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)

        #rm_scores_lst = []
        q_lst = []
        for micro_batch in micro_batches:
            with torch.no_grad():
                if self.freeze:
                    q = self._forward_micro_batch(micro_batch, prompt_length)
            #rm_scores_lst.append(rm_score)
            q_lst.append(q)
        #rm_scores = torch.concat(rm_scores_lst, dim=0)
        q = torch.concat(q_lst, dim=0)

        #rm_scores = self.prime_norm(rm_scores)

        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == q.size(0), f"{len(indices)} vs. {q.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            q = q[revert_indices]

        return (
            q.detach(),
            {
                "reward_model/raw_reward": q.sum(dim=-1).mean().item(),
            },
        )

    def update_rm(self, data: DataProto):
        # make sure we are in training mode
        self.reward_module.train()
        metrics = {}

        beta = self.config.model.get("beta_train", 0.05)

        select_keys = ["input_ids", "responses", "attention_mask", "position_ids", "seq_reward", "prompts"]

        for key in ["Q_bc", "acc_bc"]:
            if key in data.batch.keys():
                select_keys.append(key)

        batch = data.select(batch_keys=select_keys).batch
        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        dataloader = batch.split(self.config.mini_batch_size)

        rm_scores_lst = []
        q_lst = []

        for batch_idx, data in enumerate(dataloader):
            # split batch into micro_batches
            mini_batch = data
            if self.config.use_dynamic_bsz:
                max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
            else:
                micro_batches = mini_batch.split(self.config.micro_batch_size_per_gpu)
                self.gradient_accumulation = self.config.mini_batch_size // self.config.micro_batch_size_per_gpu

            self.reward_optimizer.zero_grad()

            for data in micro_batches:
                data = data.cuda()
                attention_mask = data["attention_mask"]
                acc = data["seq_reward"]

                prompt_ids = data["prompts"]
                prompt_length = prompt_ids.shape[-1]

                response_mask = attention_mask[:, prompt_length:]

                rm_score, q = self._forward_micro_batch(data, prompt_length)

                rm_scores_lst.append(rm_score)
                q_lst.append(q.detach())

                if self.config.model.loss_type == "ce":
                    dpo_loss = compute_ce_dpo_loss_rm(q, acc, response_mask=response_mask, beta=beta)
                elif self.config.model.loss_type == "dpo":
                    # the implementation of dpo is actually detached, which means we have to know the average value of w/l reward before the update.
                    dpo_loss = compute_detach_dpo_loss_rm(q, acc, Q_bc=data["Q_bc"], acc_bc=data["acc_bc"], response_mask=response_mask, beta=beta)
                elif self.config.model.loss_type == "bon_acc":
                    # change the original distribution of each sample to BoN distribution, then update reward model
                    dpo_loss = compute_detach_dpo_loss_rm(
                        q,
                        acc,
                        Q_bc=data["Q_bc"],
                        acc_bc=data["acc_bc"],
                        response_mask=response_mask,
                        beta=beta,
                        bon_mode="bon_acc",
                    )
                elif self.config.model.loss_type == "bon_rm":
                    dpo_loss = compute_detach_dpo_loss_rm(
                        q,
                        acc,
                        Q_bc=data["Q_bc"],
                        acc_bc=data["acc_bc"],
                        response_mask=response_mask,
                        beta=beta,
                        bon_mode="bon_rm",
                    )
                else:
                    raise NotImplementedError

                data = {"reward_model/dpo_loss": dpo_loss.detach().item()}

                if self.config.use_dynamic_bsz:
                    # relative to the dynamic bsz
                    loss = dpo_loss * (len(data) / self.config.ppo_mini_batch_size)
                else:
                    loss = dpo_loss / self.gradient_accumulation

                loss.backward()

                append_to_dict(metrics, data)

            grad_norm = self._optimizer_step()
            data = {"reward_model/grad_norm": grad_norm.detach().item()}
            append_to_dict(metrics, data)
        self.reward_optimizer.zero_grad()

        rm_scores = torch.cat(rm_scores_lst, dim=0)
        q = torch.concat(q_lst, dim=0)

        rm_scores = self.prime_norm(rm_scores)

        metrics.update(
            {
                "reward_model/reward": rm_scores.sum(dim=-1).mean().item(),
                "reward_model/raw_reward": q.sum(dim=-1).mean().item(),
            }
        )

        return rm_scores, metrics
