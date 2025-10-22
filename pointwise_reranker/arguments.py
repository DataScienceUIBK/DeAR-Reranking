# arguments.py
import os
from dataclasses import dataclass, field
from typing import Optional, List
from transformers import TrainingArguments

@dataclass
class ModelArguments:
    model_name_or_path: str = field(metadata={"help": "HF model id or path (student)."})
    config_name: Optional[str] = field(default=None)
    tokenizer_name: Optional[str] = field(default=None)
    cache_dir: Optional[str] = field(default=None)
    teacher_model_name_or_path: str = field(default=None, metadata={"help": "Teacher model path/id."})
    temperature: float = field(default=2.0)
    alpha: float = field(default=0.3)
    # modeling
    untie_encoder: bool = field(default=False)
    loss_type: str = field(default="softmax_ce")  # softmax_ce | ranknet_teacher | ranknet_binary | listnet_* | lambda_*
    # projection / pooler (kept for compatibility)
    add_pooler: bool = field(default=False)
    projection_in_dim: int = field(default=768)
    projection_out_dim: int = field(default=768)
    normalize: bool = field(default=False)
    # dtype hint
    dtype: Optional[str] = field(default="float32")

@dataclass
class DataArguments:
    train_dir: str = field(default=None, metadata={"help": "Path OR file to json/jsonl (optional if using load_from_disk)."})
    dataset_name: str = field(default=None, metadata={"help": "Path to HF dataset saved with save_to_disk."})
    passage_field_separator: str = field(default=' ')
    dataset_proc_num: int = field(default=12)
    train_n_passages: int = field(default=8)
    positive_passage_no_shuffle: bool = field(default=False)
    negative_passage_no_shuffle: bool = field(default=False)

    encode_in_path: List[str] = field(default=None)
    encoded_save_path: str = field(default=None)
    encode_is_qry: bool = field(default=False)
    encode_num_shard: int = field(default=1)
    encode_shard_index: int = field(default=0)
    dev_path: Optional[str] = field(default=None)

    q_max_len: int = field(default=32)
    p_max_len: int = field(default=128)
    data_cache_dir: Optional[str] = field(default=None)

    def __post_init__(self):
        if self.dataset_name is not None:
            # using load_from_disk; we still set split names for convenience
            self.dataset_split = 'train'
            self.dataset_language = 'default'
        else:
            self.dataset_name = 'json'
            self.dataset_split = 'train'
            self.dataset_language = 'default'

        if self.train_dir is not None:
            if os.path.isdir(self.train_dir):
                files = os.listdir(self.train_dir)
                self.train_dir = os.path.join(os.path.abspath(os.getcwd()), self.train_dir)
                self.train_path = [
                    os.path.join(self.train_dir, f)
                    for f in files if f.endswith(('jsonl', 'json'))
                ]
            else:
                self.train_path = [self.train_dir]
        else:
            self.train_path = None

        self.dev_split = 'dev' if self.dev_path is not None else None

from transformers import IntervalStrategy

@dataclass
class ModelTrainingArguments(TrainingArguments):
    # names kept for backwards compat with your script/flags
    warmup_ratio: float = field(default=0.1)
    negatives_x_device: bool = field(default=False)
    do_encode: bool = field(default=False)

    grad_cache: bool = field(default=False)
    gc_q_chunk_size: int = field(default=4)
    gc_p_chunk_size: int = field(default=32)
    num_questions_per_batch: int = field(default=4)
    num_answers_per_question: int = field(default=4)
    eval_interval: int = field(default=100)
    ckpt_save_interval: int = field(default=1000)
    skip_unfinished_episodes: bool = field(default=True)
    max_grad_norm: float = field(default=1.0)
