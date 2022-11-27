from typing import List

import torch
from torch.utils.data import DataLoader

from transformers import BartTokenizerFast

def get_styled_tokenizer(base_tokenizer: str, special_tokens: List[str]=None):
    tokenizer = BartTokenizerFast.from_pretrained(base_tokenizer)
    if special_tokens:
        tokenizer.add_tokens(list(special_tokens))
    style_tokens = ["<0>", "<1>"]
    tokenizer.add_tokens(style_tokens)
    style_token_ids = tokenizer.convert_tokens_to_ids(style_tokens)

    return tokenizer, style_tokens, style_token_ids

class SentenceCollator:
    def __init__(self, tokenizer, max_length, cut_first=True):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.cut_first = cut_first

    def __call__(self, samples):
        encodings = self.tokenizer(
            samples,
            return_tensors='pt',
            return_token_type_ids=False,
            padding=True,
            truncation=True,
            max_length=self.max_length + int(self.cut_first)
        )

        if self.cut_first:
            encodings = {key: value[:, 1:] for key, value in encodings.items()} # We do not use the <s> token.
        return encodings

class StyledCollator:
    def __init__(self, tokenizer, max_length, cut_first=True):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.cut_first = cut_first

    def __call__(self, samples):
        sentences, styles = zip(*samples)
        
        encodings = self.tokenizer(
            list(sentences),
            return_tensors='pt',
            return_token_type_ids=False,
            padding=True, 
            truncation=True,
            max_length=self.max_length + int(self.cut_first)
        )

        if self.cut_first:
            encodings = {key: value[:, 1:] for key, value in encodings.items()} # We do not use the <s> token.
        encodings['styles'] = torch.tensor(styles, dtype=torch.long)

        return encodings

class Loader:
    def __init__(self, loader: DataLoader, epoch: float=1.) -> None:
        self.loader = loader
        self.epoch = epoch

    def __len__(self):
        return int(len(self.loader) * self.epoch)

    def __iter__(self):
        step = 0 
        iterator = iter(self.loader)
        
        while step < len(self):
            samples = next(iterator, None)
            if samples is None:
                iterator = iter(self.loader)
                samples = next(iterator)

            step += 1
            yield samples

        return 
