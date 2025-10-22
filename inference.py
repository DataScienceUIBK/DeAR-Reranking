#reranker_inference.py
import logging
import os
import sys
from contextlib import nullcontext

from tqdm import tqdm

import torch

from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers import (
    HfArgumentParser,
)


from arguments import ModelArguments, DataArguments, \
    ModelTrainingArguments as TrainingArguments

from data import HFRerankerInferenceDataset as HFRerankDataset, RerankerInferenceDataset, RerankerInferenceCollator
from modeling import DistillRerankerModel
import time

start_time = time.time()
logger = logging.getLogger(__name__)

def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        model_args: ModelArguments
        data_args: DataArguments
        training_args: TrainingArguments

    if training_args.local_rank > 0 or training_args.n_gpu > 1:
        raise NotImplementedError('Multi-GPU encoding is not supported.')

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir
    )
    if tokenizer.name_or_path.lower().startswith("qwen") and tokenizer.bos_token_id is None:
        tokenizer.bos_token = "<|endoftext|>"
        tokenizer.bos_token_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    use_teacher =  False 
    model = DistillRerankerModel.load(
        student_model_path=model_args.model_name_or_path,
        teacher_model_path =model_args.teacher_model_name_or_path,
        use_teacher= use_teacher,
        cache_dir=model_args.cache_dir,
        torch_dtype=torch.float16 if training_args.fp16 else torch.float32,
        loss_type = model_args.loss_type
    )

    tokenizer_teacher = AutoTokenizer.from_pretrained(
        model_args.teacher_model_name_or_path,
        cache_dir=model_args.cache_dir
    )
    if tokenizer_teacher.pad_token_id is None:
        tokenizer_teacher.pad_token_id = tokenizer_teacher.eos_token_id

    if tokenizer_teacher.name_or_path.lower().startswith("qwen") and tokenizer_teacher.bos_token_id is None:
        tokenizer_teacher.bos_token = "<|endoftext|>"
        #tokenizer_teacher.pad_token_id = tokenizer_teacher.unk_token_id
    rerank_dataset = HFRerankDataset(tokenizer_student=tokenizer, tokenizer_teacher=tokenizer_teacher ,data_args=data_args, cache_dir=data_args.data_cache_dir or model_args.cache_dir)

    rerank_dataset = RerankerInferenceDataset(
            rerank_dataset.process(data_args.encode_num_shard, data_args.encode_shard_index),
            tokenizer, tokenizer_teacher, max_q_len=data_args.q_max_len, max_p_len=data_args.p_max_len
        )

    rerank_loader = DataLoader(
        rerank_dataset,
        batch_size=training_args.per_device_eval_batch_size,
        collate_fn=RerankerInferenceCollator(
            tokenizer, tokenizer_teacher
            #max_length=data_args.q_max_len+data_args.p_max_len,
            #padding='max_length'
        ),
        shuffle=False,
        drop_last=False,
        num_workers=training_args.dataloader_num_workers,
    )
    model = model.to(training_args.device)
    model.eval()
    all_results = {}
    start_time = time.time()  # Start timing inference
    for (batch_query_ids, batch_text_ids, batch, teacher) in tqdm(rerank_loader):
        with torch.cuda.amp.autocast() if training_args.fp16 else nullcontext():
            with torch.no_grad():
                for k, v in batch.items():
                    batch[k] = v.to(training_args.device)
                for k, v in teacher.items():
                    teacher[k] = v.to(training_args.device)
                model_output = model(batch, None)
                scores = model_output.scores.cpu().detach().numpy()
                if use_teacher:
                    teacher_scores = model_output.teacher_scores.cpu().detach().numpy()
                    scores = scores + teacher_scores
                else:
                    scores = scores #+ teacher_scores
                for i in range(len(scores)):
                    qid = batch_query_ids[i]
                    docid = batch_text_ids[i]
                    score = scores[i][0]
                    if qid not in all_results:
                        all_results[qid] = []
                    all_results[qid].append((docid, score))
    end_time = time.time()  # End timing inference
    print(f"Total question processing time (inference only): {end_time - start_time:.2f} seconds")

    with open(data_args.encoded_save_path, 'w') as f:
        for qid in all_results:
            results = sorted(all_results[qid], key=lambda x: x[1], reverse=True)
            for docid, score in results:
                f.write(f'{qid}\t{docid}\t{score}\n')

if __name__ == "__main__":
    main()
