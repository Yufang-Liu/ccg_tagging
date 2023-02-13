from typing import Set, List, Dict
from itertools import cycle
from collections import Counter
import argparse
import torch
from transformers import AutoTokenizer
from torch.utils.data import Dataset, get_worker_info
import numpy as np
from antu.io.vocabulary import Vocabulary
from antu.io.dataset_readers.dataset_reader import DatasetReader

# Some simple transformations to normalize tokens before encoding
# Also used in prior work: see https://github.com/nikitakit/self-attentive-parser/blob/master/src/parse_nk.py
TOKEN_MAPPING: Dict[str, str] = {
    "-LRB-": "(",
    "-RRB-": ")",
    "-LCB-": "{",
    "-RCB-": "}",
    "-LSB-": "[",
    "-RSB-": "]",
    "``": '"',
    "''": '"',
    "`": "'",
    "«": '"',
    "»": '"',
    "‘": "'",
    "’": "'",
    "“": '"',
    "”": '"',
    "„": '"',
    "‹": "'",
    "›": "'",
    "\u2013": "--",  # en dash
    "\u2014": "--",  # em dash
}


class PTBDataset(Dataset):
    def __init__(
            self,
            file_path: str,
            data_reader: DatasetReader,
            vocabulary: Vocabulary,
            cfg,
            counters=None,
            min_count=None):

        self.data = data_reader.read(file_path)
        if counters:
            for ins in self.data:
                ins.count_vocab_items(counters)
            if min_count:
                vocabulary.extend_from_counter(counters, min_count)
            else:
                vocabulary.extend_from_counter(counters)
        self.vocabulary = vocabulary
        if cfg.BERT == 'bert-base-uncased':
            self.tokenizer = AutoTokenizer.from_pretrained(cfg.BERT)
        elif cfg.BERT == 'roberta-base':
            self.tokenizer = AutoTokenizer.from_pretrained(cfg.BERT, add_prefix_space=True)
        PTBDataset.WordPAD = self.tokenizer.pad_token_id
        PTBDataset.TagPAD = self.vocabulary.get_padding_index('tag')
        self.model = cfg.DEC
        self.bert = cfg.BERT

    def __getitem__(self, idx: int):
        cleaned_words = self._preprocess(self.data[idx]['word'].tokens)
        if self.bert == 'bert-base-uncased':
            sent = '[CLS] ' + ' '.join(cleaned_words) + ' [SEP]'
            tokenized_text = self.tokenizer.tokenize(sent)
            split_mask = [1 for _ in range(len(tokenized_text))]
            sent_list = sent.split(' ')
            j = 1
            for i in range(1, len(tokenized_text)):
                if sent_list[j].lower().startswith(tokenized_text[i].lower()):
                    j += 1
                else:
                    split_mask[i] = 0
        elif self.bert == 'roberta-base':
            sent = '<s> ' + ' '.join(cleaned_words) + ' </s>'
            sent_list = sent.split(' ')
            tokenized_text = self.tokenizer.tokenize(sent_list, is_split_into_words=True)
            split_mask = [1 for _ in range(len(tokenized_text))]
            j = 1
            for i in range(1, len(tokenized_text)):
                if ord(tokenized_text[i][0]) == 288 and sent_list[j].lower().startswith(tokenized_text[i][1:].lower()):
                    j += 1
                elif ord(tokenized_text[i][0]) != 288 and sent_list[j].lower().startswith(tokenized_text[i].lower()):
                    j += 1
                else:
                    split_mask[i] = 0
        else:
            print("not used model !")
            exit(0)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)

        return indexed_tokens, split_mask, self.data[idx].index_fields(self.vocabulary), self.model

    def __len__(self):
        return len(self.data)

    def _preprocess(self, words: List[str]) -> List[str]:
        """
        Preprocess the tokens before encoding using transformers
        """
        cleaned_words: List[str] = []
        for w in words:
            w = TOKEN_MAPPING.get(w, w)
            if w == "n't" and cleaned_words != []:  # e.g., wasn't -> wasn 't
                cleaned_words[-1] = cleaned_words[-1] + "n"
                w = "'t"
            cleaned_words.append(w)
        return cleaned_words


def add_padding(batch_info):
    max_tag_len, max_token_len, max_ccg_len = 0, 0, 0
    for ins in batch_info:
        max_tag_len = max(max_tag_len, len(ins[2]['tag']['tag']))
        max_token_len = max(max_token_len, len(ins[0]))
        for i in range(len(ins[2]['tag']['tag'])):
            max_ccg_len = max(max_ccg_len, len(ins[2]['tag']['atom_ccg'][i]))

    WordPAD = PTBDataset.WordPAD
    TagPAD = PTBDataset.TagPAD
    input = {'word': [], 'tokens': [], 'tag': [], 'tag_mask': [],
             'token_mask': [], 'split': [], 'atom_tag': [], 'atom_mask': []}
    for tok, split, ins, _ in batch_info:
        pad_len = max_tag_len - len(ins['tag']['tag'])
        # word_pad_seq = [WordPAD] * pad_len
        tag_pad_seq = [TagPAD] * pad_len
        # PAD word
        # input['word'].append(ins['word']['word'] + word_pad_seq)
        # PAD tag
        input['tag'].append(ins['tag']['tag'] + tag_pad_seq)
        # add tag mask
        input['tag_mask'].append([1] * (len(ins['tag']['tag'])) + [0] * pad_len)

        pad_len = max_token_len - len(tok)
        token_pad_seq = [WordPAD] * pad_len
        # PAD token
        input['tokens'].append(tok + token_pad_seq)
        # add token mask
        input['token_mask'].append([1] * len(tok) + [0] * pad_len)
        # add split mask
        input['split'].append(split + pad_len * [0])

        sent_len = len(ins['tag']['tag'])
        atom_tag = []
        atom_mask = []
        for i in range(max_tag_len):
            if i < sent_len:
                # deal with atom_ccg
                padding_atom_length = max_ccg_len - len(ins['tag']['atom_ccg'][i])
                temp_atom = ins['tag']['atom_ccg'][i] + [TagPAD] * padding_atom_length
                temp_mask = [1] * (max_ccg_len - padding_atom_length) + [0] * padding_atom_length
            else:
                # deal with atom
                temp_atom = [TagPAD] * max_ccg_len
                temp_mask = [0] * max_ccg_len
            atom_tag.append(temp_atom)
            atom_mask.append(temp_mask)
        input['atom_tag'].append(atom_tag)
        input['atom_mask'].append(atom_mask)

    device = torch.device("cuda" if not get_worker_info() and torch.cuda.is_available() else "cpu")
    res = {}
    res['token'] = torch.tensor(input['tokens'], dtype=torch.long, device=device)
    res['tag'] = torch.tensor(input['tag'], dtype=torch.long, device=device)
    res['split'] = torch.tensor(input['split'], dtype=torch.long, device=device)
    if batch_info[0][-1] == 'LSTM':
        res['atom_tag'] = torch.tensor(input['atom_tag'], dtype=torch.long, device=device)
        res['atom_mask'] = res['atom_tag'].ne(TagPAD)
    res['token_mask'] = res['token'].ne(WordPAD)
    res['tag_mask'] = res['tag'].ne(TagPAD)
    return res
