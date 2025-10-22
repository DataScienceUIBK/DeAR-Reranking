# data.py
import os
import random
import logging
from dataclasses import dataclass
from typing import List, Tuple

import datasets
from datasets import load_from_disk, load_dataset
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, BatchEncoding, DataCollatorWithPadding

from arguments import DataArguments

logger = logging.getLogger(__name__)

class HFRerankerTrainDataset:
    def __init__(self, tokenizer_student: PreTrainedTokenizer, tokenizer_teacher: PreTrainedTokenizer,
                 data_args: DataArguments, cache_dir: str):
        data_files = data_args.train_path
        if data_files:
            data_files = {data_args.dataset_split: data_files}

        self.dataset = load_dataset(data_args.dataset_name,
                                    data_args.dataset_language,
                                    data_files=data_files, use_auth_token=None)[data_args.dataset_split]#.select(range(1))
        
        self.preprocessor = RerankerTrainPreProcessor(tokenizer_student, tokenizer_teacher,
                                                      data_args.q_max_len, data_args.p_max_len)
        self.tokenizer_student = tokenizer_student
        self.tokenizer_teacher = tokenizer_teacher
        self.q_max_len = data_args.q_max_len
        self.p_max_len = data_args.p_max_len
        self.proc_num = data_args.dataset_proc_num
        self.neg_num = data_args.train_n_passages - 1
        self.separator = getattr(self.tokenizer_student, data_args.passage_field_separator,
                                 data_args.passage_field_separator)

    def process(self, shard_num=1, shard_idx=0):
        self.dataset = self.dataset.shard(shard_num, shard_idx)
        if self.preprocessor is not None:
            self.dataset = self.dataset.map(
                self.preprocessor,
                batched=False,
                num_proc=self.proc_num,
                remove_columns=self.dataset.column_names,
                desc="Tokenizing train dataset",
            )
        return self.dataset

class RerankerTrainPreProcessor:
    def __init__(self, tokenizer_student, tokenizer_teacher, query_max_length=32, text_max_length=256, separator=' '):
        self.tokenizer_student = tokenizer_student
        self.tokenizer_teacher = tokenizer_teacher
        self.query_max_length = query_max_length
        self.text_max_length = text_max_length
        self.separator = separator

    def __call__(self, example):
        query_text = 'query: ' + example['query']
        # Tokenize query
        q_stu = self.tokenizer_student.encode(query_text, add_special_tokens=False,
                                              max_length=self.query_max_length, truncation=True)
        q_tch = self.tokenizer_teacher.encode(query_text, add_special_tokens=False,
                                              max_length=self.query_max_length, truncation=True)

        pos_stu, pos_tch = [], []
        neg_stu, neg_tch = [], []

        for pos in example['positive_passages']:
            text = (pos.get('title', '') + self.separator + pos['text']) if 'text' in pos else pos.get('title', '')
            doc_text = 'document: ' + text
            pos_stu.append(self.tokenizer_student.encode(doc_text, add_special_tokens=False,
                                                         max_length=self.text_max_length - 3, truncation=True))
            pos_tch.append(self.tokenizer_teacher.encode(doc_text, add_special_tokens=False,
                                                         max_length=self.text_max_length - 3, truncation=True))

        for neg in example['negative_passages']:
            text = (neg.get('title', '') + self.separator + neg['text']) if 'text' in neg else neg.get('title', '')
            doc_text = 'document: ' + text
            neg_stu.append(self.tokenizer_student.encode(doc_text, add_special_tokens=False,
                                                         max_length=self.text_max_length - 3, truncation=True))
            neg_tch.append(self.tokenizer_teacher.encode(doc_text, add_special_tokens=False,
                                                         max_length=self.text_max_length - 3, truncation=True))

        return {
            'query_student': q_stu, 'positives_student': pos_stu, 'negatives_student': neg_stu,
            'query_teacher': q_tch, 'positives_teacher': pos_tch, 'negatives_teacher': neg_tch
        }

class RerankerTrainDataset(Dataset):
    def __init__(self, data_args: DataArguments, dataset: datasets.Dataset,
                 tokenizer_student: PreTrainedTokenizer, tokenizer_teacher: PreTrainedTokenizer):
        self.train_data = dataset
        self.tokenizer_student = tokenizer_student
        self.tokenizer_teacher = tokenizer_teacher
        self.data_args = data_args
        self.total_len = len(self.train_data)

    def create_one_example(self, query_encoding: List[int], text_encoding: List[int], tokenizer):
        item = tokenizer.prepare_for_model(
            [tokenizer.bos_token_id] + query_encoding,
            [tokenizer.bos_token_id] + text_encoding,
            truncation='only_first',
            max_length=self.data_args.q_max_len + self.data_args.p_max_len,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        return item

    def __len__(self):
        return self.total_len

    def __getitem__(self, item) -> Tuple[List[BatchEncoding], List[BatchEncoding]]:
        group = self.train_data[item]

        # Student
        qry_stu = group['query_student']
        pos_stu = group['positives_student']
        neg_stu = group['negatives_student']

        # Teacher
        qry_tch = group['query_teacher']
        pos_tch = group['positives_teacher']
        neg_tch = group['negatives_teacher']

        feats_stu, feats_tch = [], []

        # positive
        if self.data_args.positive_passage_no_shuffle:
            p_stu = pos_stu[0]; p_tch = pos_tch[0]
        else:
            p_stu = random.sample(pos_stu, 1)[0]
            p_tch = random.sample(pos_tch, 1)[0]

        feats_stu.append(self.create_one_example(qry_stu, p_stu, self.tokenizer_student))
        feats_tch.append(self.create_one_example(qry_tch, p_tch, self.tokenizer_teacher))

        # negatives
        neg_size = self.data_args.train_n_passages - 1
        if len(neg_stu) < neg_size:
            ns_stu = random.choices(neg_stu, k=neg_size)
            ns_tch = random.choices(neg_tch, k=neg_size)
        elif self.data_args.negative_passage_no_shuffle:
            ns_stu = neg_stu[:neg_size]; ns_tch = neg_tch[:neg_size]
        else:
            ns_stu = random.sample(neg_stu, neg_size)
            ns_tch = random.sample(neg_tch, neg_size)

        for s, t in zip(ns_stu, ns_tch):
            feats_stu.append(self.create_one_example(qry_stu, s, self.tokenizer_student))
            feats_tch.append(self.create_one_example(qry_tch, t, self.tokenizer_teacher))

        return feats_stu, feats_tch

@dataclass
class RerankerTrainCollator:
    tokenizer_student: PreTrainedTokenizer
    tokenizer_teacher: PreTrainedTokenizer
    max_q_len: int = 32
    max_p_len: int = 196

    def __call__(self, features):
        student_features, teacher_features = zip(*features)
        student_collated = self.tokenizer_student.pad(
            sum(student_features, []), padding='max_length',
            max_length=self.max_q_len + self.max_p_len, return_tensors="pt"
        )
        teacher_collated = self.tokenizer_teacher.pad(
            sum(teacher_features, []), padding='max_length',
            max_length=self.max_q_len + self.max_p_len, return_tensors="pt"
        )
        return {"student": student_collated, "teacher": teacher_collated}



class HFRerankerInferenceDataset:
    def __init__(self, tokenizer_student: PreTrainedTokenizer, tokenizer_teacher: PreTrainedTokenizer, 
                 data_args: DataArguments, cache_dir: str):
        data_files = data_args.encode_in_path
        if data_files:
            data_files = {data_args.dataset_split: data_files}
        self.dataset = datasets.load_dataset(data_args.dataset_name,
                                             data_args.dataset_language,
                                             data_files=data_files, use_auth_token=None)[data_args.dataset_split]
        self.preprocessor = RerankerInferencePreProcessor(tokenizer_student, tokenizer_teacher, 
                                                          data_args.q_max_len, data_args.p_max_len)
        self.tokenizer_student = tokenizer_student
        self.tokenizer_teacher = tokenizer_teacher
        self.q_max_len = data_args.q_max_len
        self.p_max_len = data_args.p_max_len
        self.proc_num = data_args.dataset_proc_num
        self.separator = getattr(self.tokenizer_student, data_args.passage_field_separator, data_args.passage_field_separator)

    def process(self, shard_num=1, shard_idx=0):
        self.dataset = self.dataset.shard(shard_num, shard_idx)
        if self.preprocessor is not None:
            self.dataset = self.dataset.map(
                self.preprocessor,
                batched=False,
                num_proc=self.proc_num,
                remove_columns=self.dataset.column_names,
                desc="Running tokenizer on inference dataset",
            )
        return self.dataset


class RerankerInferencePreProcessor:
    def __init__(self, tokenizer_student, tokenizer_teacher, query_max_length=32, text_max_length=256, separator=' '):
        self.tokenizer_student = tokenizer_student
        self.tokenizer_teacher = tokenizer_teacher
        self.query_max_length = query_max_length
        self.text_max_length = text_max_length
        self.separator = separator

    def __call__(self, example):
        query_text = 'query: ' + example['query'] 
        doc_text = 'document: ' + example.get('title', '') + self.separator + example['text']
        # Tokenize query and document separately for student & teacher
        query_student = self.tokenizer_student.encode(query_text, 
                                                      add_special_tokens=False, 
                                                      max_length=self.query_max_length, 
                                                      truncation=True)
        query_teacher = self.tokenizer_teacher.encode(query_text, 
                                                      add_special_tokens=False, 
                                                      max_length=self.query_max_length, 
                                                      truncation=True)

        text_student = self.tokenizer_student.encode(doc_text, 
                                                     add_special_tokens=False, 
                                                     max_length=self.text_max_length-3, 
                                                     truncation=True)
        text_teacher = self.tokenizer_teacher.encode(doc_text, 
                                                     add_special_tokens=False, 
                                                     max_length=self.text_max_length-3, 
                                                     truncation=True)

        return {
            'query_id': example['query_id'],
            'query_student': query_student, 'text_student': text_student,
            'query_teacher': query_teacher, 'text_teacher': text_teacher,
            'text_id': example['docid']
        }


class RerankerInferenceDataset(Dataset):
    input_keys = ['query_id', 'query_student', 'text_id', 'text_student', 'query_teacher', 'text_teacher']

    def __init__(self, dataset: datasets.Dataset, tokenizer_student: PreTrainedTokenizer, tokenizer_teacher: PreTrainedTokenizer, max_q_len=32, max_p_len=256):
        self.encode_data = dataset
        self.tokenizer_student = tokenizer_student
        self.tokenizer_teacher = tokenizer_teacher
        self.max_q_len = max_q_len
        self.max_p_len = max_p_len

    def __len__(self):
        return len(self.encode_data)

    def __getitem__(self, item) -> Tuple[str, str, BatchEncoding, BatchEncoding]:
        query_id, query_student, text_id, text_student, query_teacher, text_teacher = (
            self.encode_data[item][f] for f in self.input_keys
        )

        # Encode student input
        encoded_pair_student = self.tokenizer_student.prepare_for_model(
            [self.tokenizer_student.bos_token_id] + query_student,
            [self.tokenizer_student.bos_token_id] + text_student,
            truncation='only_second',  # prevents warning
            padding='max_length',
            max_length=self.max_q_len + self.max_p_len,
            return_token_type_ids=False,
        )

        # Encode teacher input
        encoded_pair_teacher = self.tokenizer_teacher.prepare_for_model(
            [self.tokenizer_teacher.bos_token_id] + query_teacher,
            [self.tokenizer_teacher.bos_token_id] + text_teacher,
            truncation='only_second',  # prevents warning
            padding='max_length',
            max_length=self.max_q_len + self.max_p_len,
            return_token_type_ids=False,
        )

        return query_id, text_id, encoded_pair_student, encoded_pair_teacher

@dataclass
class RerankerInferenceCollator(DataCollatorWithPadding):
    def __init__(self, tokenizer_student: PreTrainedTokenizer, tokenizer_teacher: PreTrainedTokenizer, 
                 max_q_len: int = 32, max_p_len: int = 196):
        self.tokenizer_student = tokenizer_student
        self.tokenizer_teacher = tokenizer_teacher
        self.max_q_len = max_q_len
        self.max_p_len = max_p_len

    def __call__(self, features):
        query_ids = [x[0] for x in features]
        text_ids = [x[1] for x in features]
        student_features = [x[2] for x in features]
        teacher_features = [x[3] for x in features]

        # Collate student data
        student_collated = self.tokenizer_student.pad(
            student_features,
            padding='max_length',
            max_length=self.max_q_len + self.max_p_len,
            return_tensors="pt",
        )

        # Collate teacher data
        teacher_collated = self.tokenizer_teacher.pad(
            teacher_features,
            padding='max_length',
            max_length=self.max_q_len + self.max_p_len,
            return_tensors="pt",
        )
        return query_ids, text_ids, student_collated, teacher_collated
