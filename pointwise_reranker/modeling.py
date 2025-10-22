# modeling.py
from dataclasses import dataclass
from typing import Dict, Optional

import logging
import torch
import torch.nn.functional as F
from torch import nn, Tensor

from transformers import AutoModelForSequenceClassification, PreTrainedModel
from transformers.utils import ModelOutput

from peft import PeftConfig, PeftModel, get_peft_model, LoraConfig, TaskType

from arguments import ModelArguments, ModelTrainingArguments as TrainingArguments
from rank_loss import RankLoss

logger = logging.getLogger(__name__)

@dataclass
class RerankerOutput(ModelOutput):
    loss: Optional[Tensor] = None
    scores: Optional[Tensor] = None
    teacher_scores: Optional[Tensor] = None

class DistillRerankerModel(nn.Module):
    def __init__(
        self,
        student_model: PeftModel,
        teacher_model: Optional[PreTrainedModel],
        train_batch_size: Optional[int] = None,
        temperature: float = 2.0,
        alpha: float = 0.5,
        loss_type: str = "softmax_ce",
        use_teacher: bool = True,
    ):
        super().__init__()
        self.student = student_model
        self.teacher = teacher_model
        self.config = student_model.config
        self.train_batch_size = train_batch_size
        self.temperature = temperature
        self.alpha = alpha
        self.loss_type = loss_type
        self.use_teacher = use_teacher

        if self.teacher is not None and self.use_teacher:
            for p in self.teacher.parameters():
                p.requires_grad = False

        self.cross_entropy = nn.CrossEntropyLoss(reduction="mean")
        print(f"[DistillRerankerModel] loss_type={loss_type}, T={temperature}, alpha={alpha}")

    def forward(self, pair_student: Dict[str, Tensor], pair_teacher: Optional[Dict[str, Tensor]] = None):
        # student logits: (B*N, 1)
        student_logits = self.student(**pair_student, return_dict=True).logits  # (batch*passages, 1)

        if self.train_batch_size:
            with torch.no_grad():
                teacher_logits = None
                if self.use_teacher and pair_teacher is not None:
                    teacher_logits = self.teacher(**pair_teacher, return_dict=True).logits  # (B*N, 1)

            grouped_student = student_logits.view(self.train_batch_size, -1)  # (B, N)
            grouped_teacher = teacher_logits.view(self.train_batch_size, -1) if teacher_logits is not None else None

            # main supervised ranking loss
            if self.loss_type == "ranknet_teacher":
                relevance = grouped_teacher.detach()
                main_loss = RankLoss.rank_net(grouped_student, relevance)
            elif self.loss_type == "ranknet_binary":
                relevance = torch.zeros_like(grouped_student)
                relevance[:, 0] = 1
                main_loss = RankLoss.rank_net(grouped_student, relevance)
            elif self.loss_type == "softmax_ce":
                target = torch.zeros(self.train_batch_size, dtype=torch.long, device=grouped_student.device)
                main_loss = self.cross_entropy(grouped_student, target)
            elif self.loss_type == "listnet_teacher":
                relevance = grouped_teacher.detach()
                main_loss = RankLoss.list_net(grouped_student, relevance)
            elif self.loss_type == "lambda_loss_teacher":
                relevance = grouped_teacher.detach()
                main_loss = RankLoss.lambda_loss(grouped_student, relevance, k=5, weighing_scheme="ndcgLoss2_scheme")
            elif self.loss_type == "listnet_binary":
                relevance = torch.zeros_like(grouped_student); relevance[:, 0] = 1
                main_loss = RankLoss.list_net(grouped_student, relevance)
            elif self.loss_type == "lambda_loss_binary":
                relevance = torch.zeros_like(grouped_student); relevance[:, 0] = 1
                main_loss = RankLoss.lambda_loss(grouped_student, relevance, k=5, weighing_scheme="ndcgLoss2_scheme")
            elif self.loss_type == "ranknet_binary_softmax_ce":
                relevance = torch.zeros_like(grouped_student); relevance[:, 0] = 1
                loss_rank = RankLoss.rank_net(grouped_student, relevance)
                target = torch.zeros(self.train_batch_size, dtype=torch.long, device=grouped_student.device)
                loss_ce = self.cross_entropy(grouped_student, target)
                beta = 0.5
                main_loss = beta * loss_rank + (1 - beta) * loss_ce
            else:
                raise ValueError(f"Unsupported loss type: {self.loss_type}")

            # KD loss (requires teacher)
            if grouped_teacher is not None:
                s = F.log_softmax(grouped_student / self.temperature, dim=1)
                t = F.softmax(grouped_teacher / self.temperature, dim=1)
                kl = F.kl_div(s, t, reduction="batchmean") * (self.temperature ** 2)
                total = (1 - self.alpha) * main_loss + self.alpha * kl
            else:
                total = main_loss

            return RerankerOutput(loss=total, scores=student_logits, teacher_scores=teacher_logits)

        # inference mode
        tlogits = None
        if self.use_teacher and pair_teacher is not None:
            with torch.no_grad():
                tlogits = self.teacher(**pair_teacher, return_dict=True).logits
        return RerankerOutput(loss=None, scores=student_logits, teacher_scores=tlogits)

    def gradient_checkpointing_enable(self):
        # match llama base structures
        if hasattr(self.student, "base_model") and hasattr(self.student.base_model, "model"):
            self.student.base_model.model.gradient_checkpointing_enable()

    @classmethod
    def load(
        cls,
        student_model_path,
        teacher_model_path=None,
        temperature=2.0,
        alpha=0.3,
        use_teacher=False,
        loss_type="softmax_ce",
        **hf_kwargs,
    ):
        config = PeftConfig.from_pretrained(student_model_path)
        student_base = AutoModelForSequenceClassification.from_pretrained(
            config.base_model_name_or_path, num_labels=1, **hf_kwargs
        )
        if student_base.config.pad_token_id is None:
            student_base.config.pad_token_id = 0
        student = PeftModel.from_pretrained(student_base, student_model_path)
        student = student.merge_and_unload()

        teacher = None
        if use_teacher and teacher_model_path:
            teacher = AutoModelForSequenceClassification.from_pretrained(
                teacher_model_path, num_labels=1, **hf_kwargs
            )
            if teacher.config.pad_token_id is None:
                teacher.config.pad_token_id = 0

        return cls(
            student_model=student,
            teacher_model=teacher,
            temperature=temperature,
            alpha=alpha,
            use_teacher=use_teacher,
            loss_type=loss_type
        )

    @classmethod
    def build(cls, model_args: ModelArguments, train_args: TrainingArguments, temperature=2.0, alpha=0.5, **hf_kwargs):
        student_base = AutoModelForSequenceClassification.from_pretrained(
            model_args.model_name_or_path, num_labels=1, **hf_kwargs
        )
        if train_args.gradient_checkpointing:
            student_base.enable_input_require_grads()
        if student_base.config.pad_token_id is None:
            student_base.config.pad_token_id = 0

        peft_cfg = LoraConfig(
            base_model_name_or_path=model_args.model_name_or_path,
            task_type=TaskType.SEQ_CLS,
            r=32,
            lora_alpha=64,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj", "o_proj", "down_proj", "up_proj", "gate_proj"],
            inference_mode=False,
        )
        student = get_peft_model(student_base, peft_cfg)

        teacher = AutoModelForSequenceClassification.from_pretrained(
            model_args.teacher_model_name_or_path, num_labels=1, **hf_kwargs
        )
        if teacher.config.pad_token_id is None:
            teacher.config.pad_token_id = 0

        model = cls(
            student_model=student,
            teacher_model=teacher,
            train_batch_size=train_args.per_device_train_batch_size,
            temperature=temperature,
            alpha=alpha,
            loss_type=model_args.loss_type,
            use_teacher=True,
        )
        # deepspeed compat
        for m in model.modules():
            if not hasattr(m, "ds_grads_remaining"):
                m.ds_grads_remaining = 0
        return model

    def save(self, output_dir: str):
        self.student.save_pretrained(output_dir)
