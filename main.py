"""
Huggingface compatible implementation of Mamba.

# FIXME: this implementation seems to require 25% more memory than the orig.py...
"""

import argparse
from functools import partial
import math
from pprint import pprint
import random
import os
import sys
from typing import Optional

from datasets import load_dataset
import evaluate
import numpy as np
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    DataCollatorWithPadding,
    PretrainedConfig,
    PreTrainedModel,
    Trainer,
    TrainingArguments,
)
from transformers.modeling_outputs import (
    CausalLMOutput,
    MaskedLMOutput,
    SequenceClassifierOutput,
)
from transformers.trainer_utils import EvalPrediction
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss
from torch import LongTensor, Tensor

from mamba_ssm.modules.mamba_simple import Block
from mamba_ssm.models.mixer_seq_simple import create_block
try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

sys.path.insert(0, "/home/lk3591/Documents/code/RawByteClf")

from src.utils import count_parameters
from src.data.loaders_pt import get_bodmas_dataset, get_goodware_vs_malware_dataset, preprocess_fn_shift_token_idx, preprocess_fn_add_cls_token
from src.learn.tokenization import get_tokenizer
from src.learn.utils import get_mem



random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


ACCURACY = evaluate.load("accuracy")
F1 = evaluate.load("f1")


def clf_compute_metrics(eval_pred: EvalPrediction) -> dict[str, float]:
    predictions, labels = eval_pred.predictions, eval_pred.label_ids
    # predictions = predictions[0]  # its unclear exactly why, but the predictions are wrapped in a tuple

    predictions = np.argmax(predictions, axis=1)
    metrics = {
        "accuracy": ACCURACY.compute(predictions=predictions, references=labels)["accuracy"],
        "f1-macro": F1.compute(predictions=predictions, references=labels, average="macro")["f1"],
        "f1-micro": F1.compute(predictions=predictions, references=labels, average="micro")["f1"],
    }
    return metrics


class MambaConfig(PretrainedConfig):

    def __init__(
        self,
        d_model: int = 2560,
        n_layer: int = 64,
        vocab_size: int = 50277,
        ssm_cfg: Optional[dict] = None,
        rms_norm: bool = True,
        residual_in_fp32: bool = True,
        fused_add_norm: bool = True,
        pad_vocab_size_multiple: int = 8,
        initializer_range: float = 0.02,
        rescale_prenorm_residual: bool = True,
        n_residuals_per_layer: int = 1,
        norm_epsilon: float = 1e-5,
        pad_token_id: int = -1,
        mlp_hidden_size: int = 512,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.d_model = d_model
        self.n_layer = n_layer
        if vocab_size % pad_vocab_size_multiple != 0:
            vocab_size += pad_vocab_size_multiple - (vocab_size % pad_vocab_size_multiple)
        self.vocab_size = vocab_size
        self.ssm_cfg = {} if ssm_cfg is None else ssm_cfg
        self.rms_norm = rms_norm
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.pad_vocab_size_multiple = pad_vocab_size_multiple
        self.initializer_range = initializer_range
        self.rescale_prenorm_residual = rescale_prenorm_residual
        self.n_residuals_per_layer = n_residuals_per_layer
        self.norm_epsilon = norm_epsilon
        self.pad_token_id = pad_token_id
        self.mlp_hidden_size = mlp_hidden_size
        self.pruned_heads = None  # compatibility with transformers


class MambaPreTrainedModel(PreTrainedModel):

    config_class = MambaConfig
    load_tf_weights = None
    base_model_prefix = "backbone"
    supports_gradient_checkpointing = True
    config: MambaConfig

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            if module.bias is not None:
                if not getattr(module.bias, "_no_reinit", False):
                    nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=self.config.initializer_range)

        if self.config.rescale_prenorm_residual:
            for name, p in module.named_parameters():
                if name in ["out_proj.weight", "fc2.weight"]:
                    nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                    with torch.no_grad():
                        p /= math.sqrt(self.config.n_residuals_per_layer * self.config.n_layer)


class MixerModel(MambaPreTrainedModel):

    def __init__(self, config: MambaConfig) -> None:
        super().__init__(config)

        self.embedding = nn.Embedding(
            config.vocab_size,
            config.d_model,
        )

        if config.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        self.layers: list[Block] = nn.ModuleList(
            [
                create_block(
                    config.d_model,
                    ssm_cfg=config.ssm_cfg,
                    norm_epsilon=config.norm_epsilon,
                    rms_norm=config.rms_norm,
                    residual_in_fp32=config.residual_in_fp32,
                    fused_add_norm=config.fused_add_norm,
                    layer_idx=i,
                )
                for i in range(config.n_layer)
            ]
        )

        self.norm_f = (nn.LayerNorm if not config.rms_norm else RMSNorm)(
            config.d_model,
            eps=config.norm_epsilon,
        )
        self.post_init()

    def allocate_inference_cache(self, batch_size, max_seqlen, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    def forward(self, input_ids: LongTensor):
        hidden_states = self.embedding(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(hidden_states, residual)
        if not self.config.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.config.residual_in_fp32,
            )
        return hidden_states


class MambaLMHeadModel(MambaPreTrainedModel):

    def __init__(self, config: MambaConfig) -> None:
        super().__init__(config)
        self.backbone = MixerModel(config)
        self.head = nn.Linear(
            config.d_model,
            config.vocab_size,
            bias=False,
        )
        # FIXME: should the weights be tied in another manner?
        self.tie_weights()
        self.post_init()

    def tie_weights(self):
        self.head.weight = self.backbone.embedding.weight

    def allocate_inference_cache(self, batch_size, max_seqlen, **kwargs):
        return self.backbone.allocate_inference_cache(batch_size, max_seqlen, **kwargs)

    def forward(self, input_ids: LongTensor, labels: Optional[LongTensor] = None) -> CausalLMOutput:
        hidden_states: Tensor = self.backbone(input_ids)
        logits: Tensor = self.head(hidden_states)

        loss = None
        labels = input_ids
        if labels is not None:
            labels = labels.to(logits.device)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return CausalLMOutput(loss=loss, logits=logits, hidden_states=hidden_states)


class MambaForSequenceClassification(MambaPreTrainedModel):

    def __init__(self, config: MambaConfig) -> None:
        super().__init__(config)
        self.backbone = MixerModel(config)
        self.head = nn.Sequential(
            nn.Linear(config.d_model, config.mlp_hidden_size),
            nn.LeakyReLU(),
            nn.Linear(config.mlp_hidden_size, config.num_labels),
        )

        self.num_labels = config.num_labels
        self.post_init()

    def forward(self, input_ids: LongTensor, labels: Optional[LongTensor] = None) -> CausalLMOutput:
        hidden_states: Tensor = self.backbone(input_ids)
        logits: Tensor = self.head(hidden_states)

        batch_size = input_ids.shape[0]
        sequence_lengths = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
        sequence_lengths = sequence_lengths % input_ids.shape[-1]
        sequence_lengths = sequence_lengths.to(logits.device)

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=pooled_logits,
            # hidden_states=hidden_states,
            # logits=None,  # These will cause OOM errors during evaluation.
            hidden_states=None,
        )


parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="nlp-clm", choices=["nlp-clf", "nlp-clm", "mal-clf", "mal-clm"])
parser.add_argument("--max_length", type=int, default=16384)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
args = parser.parse_args()


num_classes = None
id2label = None
label2id = None

m = get_mem(unit="GB")
print(f"MEMORY: mem_used={m[2]}, mem_avail={m[1]}, mem_total={m[0]}", flush=True)

if args.task in ["nlp-clf", "nlp-clm"]:

    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    tokenizer.model_input_names.remove("attention_mask")
    tokenizer.add_special_tokens({"pad_token": "<|endoftext|>"})
    vocab_size = len(tokenizer)

    dataset = load_dataset("ag_news")
    dataset = dataset.map(
        lambda examples: tokenizer(
            examples["text"],
            truncation=True,
            max_length=args.max_length,
        ),
        batched=True,
    )


    dataset["tr"] = dataset.pop("train")
    dataset["vl"] = dataset.pop("test")

    dataset = dataset.rename_column("label", "labels")
    if args.task == "nlp-clm":
        dataset = dataset.remove_columns("labels")
    elif args.task == "nlp-clf":
        num_classes = dataset["tr"].info.features["labels"].num_classes
        id2label = {i: l for i, l in enumerate(dataset["tr"].info.features["labels"].names)}
        label2id = {l: i for i, l in enumerate(id2label.values())}

    print(f"{dataset=}")
    print(f"{dataset['tr']=}")

elif args.task in ["mal-clf", "mal-clm"]:
    tokenizer = get_tokenizer(False)
    tokenizer.model_input_names.remove("attention_mask")
    vocab_size = 264

    fn_1 = partial(preprocess_fn_shift_token_idx, shift=len(tokenizer.all_special_ids))
    fn_2 = partial(preprocess_fn_add_cls_token, cls_token_id=tokenizer.cls_token_id)
    preprocess_fn = lambda x: fn_2(fn_1(x))

    # dataset = get_goodware_vs_malware_dataset(
    #     n_mal=10000, n_ben=10000, max_length=args.max_length, preprocess_fn=preprocess_fn
    # )
    # num_classes = 2
    # id2label = {0: "benign", 1: "malware"}
    # label2id = {"benign": 0, "malware": 1}

    dataset = get_bodmas_dataset(top_k=10, max_length=args.max_length, preprocess_fn=preprocess_fn)[0]
    id2label = dataset["tr"].dataset.id2label
    label2id = dataset["tr"].dataset.label2id
    num_classes = dataset["tr"].dataset.num_classes


    print(f"{dataset=}")
    try:
        print(f"{dataset['tr'].dataset.datasets[0]=}")
    except Exception:
        print(f"{dataset['tr'].dataset=}")

    print(f"{id2label=}")
    print(f"{num_classes=}")


config = MambaConfig(
    d_model=512,
    n_layer=2,
    vocab_size=vocab_size,
    ssm_cfg={},
    pad_token_id=tokenizer.pad_token_id,
    num_classes=num_classes,
    id2label=id2label,
    label2id=label2id,
)
print(f"{config=}")


if args.task in ["nlp-clm", "mal-clm"]:
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, pad_to_multiple_of=8)
    model = MambaLMHeadModel(config)
elif args.task in ["nlp-clf", "mal-clf"]:
    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    model = MambaForSequenceClassification(config)

print(f"{data_collator=}")
print(f"{model=}")
print(f"{count_parameters(model, requires_grad=False)=}")

args = TrainingArguments(
    output_dir=f"./output/{args.task}/{args.max_length}",
    overwrite_output_dir=True,
    do_train=True,
    do_eval=True,

    evaluation_strategy="steps",
    save_strategy="steps",
    logging_steps=10,
    eval_steps=100,
    save_steps=1000,
    save_total_limit=3,

    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    eval_accumulation_steps=None,

    learning_rate=1e-4,
    lr_scheduler_type="inverse_sqrt",
    warmup_steps=100,
    weight_decay=0.01,
    max_grad_norm=1.0,
    optim="adamw_torch",

    num_train_epochs=25,
    auto_find_batch_size=False,
    fp16=True,
    fp16_full_eval=True,
)

trainer = Trainer(
    model=model,
    train_dataset=dataset["tr"],
    eval_dataset=dataset["vl"],
    tokenizer=tokenizer,
    args=args,
    data_collator=data_collator,
    compute_metrics=clf_compute_metrics,
)

print("Training...")
m = get_mem(unit="GB")
print(f"MEMORY: mem_used={m[2]}, mem_avail={m[1]}, mem_total={m[0]}", flush=True)
trainer.train()
