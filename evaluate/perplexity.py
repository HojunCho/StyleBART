import math
from statistics import fmean as mean

from torch.utils.data import DataLoader

from transformers import (
    GPT2TokenizerFast,
    GPT2LMHeadModel
)

import hydra
from tqdm import tqdm

from style_bart.data import utils as data_utils

def perplexity(sentences, model, tokenizer, batch_size=256, num_workers=4):
    # Data loader
    collator = data_utils.SentenceCollator(tokenizer, model.config.max_length, cut_first=False)
    loader = DataLoader(
        sentences, 
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True
    )

    losses = []
    for encodings in tqdm(loader, desc='Perplexity'):
        encodings = {k: v.cuda(non_blocking=True) for k, v in encodings.items()}
        encodings['labels'] = encodings['input_ids'].masked_fill((1-encodings['attention_mask']).bool(), -100)
        outputs = model(**encodings)

        losses.append(outputs.loss.item())

        del encodings, outputs
    
    losses = list(filter(math.isfinite, losses))
    return math.exp(mean(losses))

@hydra.main("../config/evaluate", "eval_lm", None)
def main(config):
    # Dataset
    with open(config.target) as fd:
        dataset = [sentence.strip() for sentence in fd.readlines()]

    tokenizer = GPT2TokenizerFast.from_pretrained(config.model.tokenizer, pad_token='<pad>') 

    model = GPT2LMHeadModel.from_pretrained(config.model.pretrained, pad_token_id=tokenizer.pad_token_id)
    model.resize_token_embeddings(len(tokenizer))
    model.cuda().eval()

    print(perplexity(dataset, model, tokenizer, config.eval.batch_size, config.eval.num_workers))

if __name__ == "__main__":
    main()