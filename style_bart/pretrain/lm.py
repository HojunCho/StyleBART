import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
from statistics import fmean as mean

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.optim import AdamW

from transformers import (
    GPT2Config,
    BartForConditionalGeneration
)

import hydra
from tqdm import tqdm

from style_bart import models, utils
from style_bart.data import datasets, utils as data_utils

@hydra.main("../../config", "pretrain_lm", None)
def main(config):
    utils.set_seed(config.seed + 1)

    # Dataset
    train_dataset = datasets.get_sentences(config.data, 'train', config.label)
    dev_dataset = datasets.get_sentences(config.data, 'dev', config.label)

    # Tokenizer and model
    tokenizer, _, _ = data_utils.get_styled_tokenizer(
        config.model.tokenizer, 
        config.model.get('additional_tokens', None)
    )

    if isinstance(config.model.arch, str):
        assert config.model.pretrained
        encoder = BartForConditionalGeneration.from_pretrained(config.model.arch).model.encoder
        input_ids=torch.tensor(
            [[tokenizer.bos_token_id, tokenizer.mask_token_id, tokenizer.eos_token_id]], 
            dtype=torch.long
        )
        with torch.no_grad():
            encoder_hidden_states = encoder(input_ids)[0]

        del input_ids, encoder
        
        lm = models.BartForCausalLM.from_pretrained(
            config.model.arch,
            max_length=config.model.max_length,
            tie_word_embeddings=True,
            forced_bos_token_id=None,
            encoder_hidden_states=encoder_hidden_states
        )
        lm.resize_token_embeddings(len(tokenizer))
    else:
        lm = models.GPT2LMHeadModel(
            GPT2Config(
                vocab_size=len(tokenizer),
                max_length=config.model.max_length,
                **config.model.arch
            )
        )

    lm.cuda()

    # Data loader
    batch_size = config.train.batch_size
    accumulation = config.train.accumulation

    collator = data_utils.SentenceCollator(tokenizer, config.model.max_length)
    train_loader = data_utils.Loader(DataLoader(
        train_dataset, 
        batch_size=batch_size // accumulation,
        shuffle=True,
        num_workers=config.train.num_workers,
        collate_fn=collator,
        pin_memory=True
    ), 100.)

    dev_loader = DataLoader(
        dev_dataset,
        batch_size=batch_size // accumulation,
        shuffle=False,
        num_workers=config.train.num_workers,
        collate_fn=collator,
        pin_memory=True
    )

    # Optimizer
    optimizer = AdamW(
        lm.parameters(), 
        lr=config.train[
            'learning_rate' \
            if isinstance(config.model.arch, str) \
            else 'learning_rate_init'
        ]
    )

    # Tensorboard & Evaluation
    writer = SummaryWriter(log_dir='review')

    @torch.no_grad()
    def evaluate(lm, loader):
        lm.eval()
        losses = []
        for encodings in tqdm(loader, desc='Evaluate LM'):
            encodings = {k: v.cuda(non_blocking=True) for k, v in encodings.items()}
            encodings['labels'] = encodings['input_ids']
            outputs = lm(**encodings)
            losses.append(outputs.loss.item())

            del encodings, outputs
        return mean(losses)

    last_eval_loss = evaluate(lm, dev_loader)
    writer.add_scalar(f"LM/{config.label}/loss", last_eval_loss, global_step=0)

    # Train
    step_loss = 0
    endurance_count = 0
    lm.train()
    for iteration, encodings in enumerate(tqdm(train_loader, desc='Pratrain LM')):
        step = iteration // accumulation + 1
        encodings = {k: v.cuda(non_blocking=True) for k, v in encodings.items()}

        # Forward & Backward
        encodings['labels'] = encodings['input_ids'].masked_fill((1 - encodings['attention_mask']).bool(), -100)
        
        outputs = lm(**encodings)

        loss = outputs.loss / accumulation
        loss.backward()
        step_loss += loss.item()

        del encodings, outputs, loss

        if (iteration + 1) % accumulation:
            continue

        # Optimization
        clip_grad_norm_(lm.parameters(), max_norm=config.train.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        step += 1
        writer.add_scalar(f"LM/{config.label}/train_loss", step_loss, global_step=step)
        step_loss = 0

        if step % config.train.eval_per_step:
            continue

        # Evaluation
        eval_loss = evaluate(lm, dev_loader)
        writer.add_scalar(f"LM/{config.label}/loss", eval_loss, global_step=step)

        if eval_loss > last_eval_loss:
            if endurance_count < config.train.endurance:
                endurance_count += 1
            else:
                break
        else:
            endurance_count = 0
            last_eval_loss = eval_loss
            lm.save_pretrained(f"dump")

        lm.train()

if __name__ == "__main__":
    main()