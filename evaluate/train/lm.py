import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import math

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.optim import AdamW

from transformers import (
    GPT2TokenizerFast, 
    GPT2LMHeadModel,
)

import hydra
from tqdm import tqdm

from style_bart import utils
from style_bart.data import datasets, utils as data_utils
from .. import perplexity

@hydra.main("../../config/evaluate", "train_lm", None)
def main(config):
    utils.set_seed(config.seed)

    # Dataset
    train_dataset = datasets.get_sentences(config.data, 'train', label=config.label)
    dev_dataset = datasets.get_sentences(config.data, 'dev', label=config.label)

    # Tokenizer and model
    tokenizer = GPT2TokenizerFast.from_pretrained(config.model.tokenizer, pad_token='<pad>') 

    lm = GPT2LMHeadModel.from_pretrained(config.model.pretrained, pad_token_id=tokenizer.pad_token_id)
    lm.resize_token_embeddings(len(tokenizer))
    lm.cuda()

    # Data loader
    batch_size = config.train.batch_size
    accumulation = config.train.accumulation

    collator = data_utils.SentenceCollator(tokenizer, lm.config.max_length, cut_first=False)
    train_loader = data_utils.Loader(DataLoader(
        train_dataset, 
        batch_size=batch_size // accumulation,
        shuffle=True,
        num_workers=config.train.num_workers,
        collate_fn=collator,
        pin_memory=True
    ), 100.)

    # Optimizer
    optimizer = AdamW(lm.parameters(), lr=config.train.learning_rate)

    # Tensorboard & Initial evaluation
    writer = SummaryWriter(log_dir='review')
    ppl = perplexity.perplexity(dev_dataset, lm, tokenizer)
    loss = math.log(ppl)
    writer.add_scalar(f"LM/{config.label}/loss", loss, global_step=0)
    writer.add_scalar(f"LM/{config.label}/perplexity", ppl, global_step=0)
    last_eval_loss = loss

    # Train
    step_loss = 0
    endurance_count = 0
    lm.train()
    for iteration, encodings in enumerate(tqdm(train_loader, desc='Train')):
        step = iteration // accumulation + 1
        encodings = {k: v.cuda(non_blocking=True) for k, v in encodings.items()}

        # Forward & Backward
        encodings['labels'] = encodings['input_ids'].masked_fill((1-encodings['attention_mask']).bool(), -100)
        outputs = lm(**encodings)
        loss = outputs.loss / accumulation
        loss.backward()

        step_loss += loss.item()

        del encodings, outputs, loss
        
        if (iteration + 1) % accumulation:
            continue

        # Optimization
        clip_grad_norm_(lm.parameters(), max_norm=1.)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        writer.add_scalar(f"LM/{config.label}/train_loss", step_loss, global_step=step)
        step_loss = 0

        if step % config.train.eval_per_step:
            continue

        # Evaluation
        ppl = perplexity.perplexity(dev_dataset, lm, tokenizer)
        loss = math.log(ppl)
        writer.add_scalar(f"LM/{config.label}/loss", loss, global_step=step)
        writer.add_scalar(f"LM/{config.label}/perplexity", ppl, global_step=step)

        if loss > last_eval_loss:
            if endurance_count < config.train.endurance:
                endurance_count += 1
            else:
                break
        else:
            endurance_count = 0
            last_eval_loss = loss
            lm.save_pretrained("dump")

        lm.train()

if __name__ == "__main__":
    main()