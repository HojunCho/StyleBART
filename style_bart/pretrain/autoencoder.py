import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
from itertools import count

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.optim import AdamW

from transformers import BartConfig

import hydra
from tqdm import tqdm

from style_bart import models, utils
from style_bart.data import datasets, utils as data_utils

@hydra.main("../../config", "pretrain_autoencoder", None)
def main(config):
    utils.set_seed(config.seed + 2)

    # Dataset
    train_dataset = datasets.StyledSentences(config.data, 'train')

    # Tokenizer and model
    tokenizer, _, style_token_ids = data_utils.get_styled_tokenizer(
        config.model.tokenizer, 
        config.model.get('additional_tokens', None)
    )

    if isinstance(config.model.arch, str):
        if config.model.pretrained:
            model = models.StyleBart.from_pretrained(
                config.model.arch,
                max_length=config.model.max_length,
                tie_word_embeddings=True,
                forced_bos_token_id=None,
                num_beams=1,
                min_length=1,
                style_token_ids=style_token_ids,
            )
        else:
            model = models.StyleBart(
                BartConfig.from_pretrained(
                    config.model.arch,
                    max_length=config.model.max_length,
                    tie_word_embeddings=True,
                    forced_bos_token_id=None,
                    num_beams=1,
                    min_length=1,
                ),
                style_token_ids=style_token_ids
            )

        model.resize_token_embeddings(len(tokenizer))
        model.get_input_embeddings().weight.data[style_token_ids, :] \
            = model.get_input_embeddings().weight.data[[model.config.decoder_start_token_id], :]

    else:
        model = models.StyleBart(
            BartConfig(
                vocab_size=len(tokenizer),
                tie_word_embeddings=True,
                max_length=config.model.max_length,
                **config.model.arch
            ),
            style_token_ids=style_token_ids
        )

    model.cuda()

    # Data loader
    batch_size = config.train.batch_size
    accumulation = config.train.accumulation

    collator = data_utils.StyledCollator(tokenizer, config.model.max_length)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size // accumulation,
        shuffle=True,
        num_workers=config.train.num_workers,
        collate_fn=collator,
        pin_memory=True
    )

    # Optimizer
    optimizer = AdamW(
        model.parameters(), 
        lr=config.train[
            'learning_rate' \
            if isinstance(config.model.arch, str) and config.model.pretrained \
            else 'learning_rate_init'
        ]
    )

    # Tensorboard
    writer = SummaryWriter(log_dir='review')

    # Train
    model.train()
    train_iterator = iter(train_loader)

    step_loss = 0
    counts = 0
    for iteration in tqdm(count(0), desc='Pratrain Model'):
        step = iteration // accumulation + 1
        encodings = next(train_iterator, None)
        if encodings is None:
            train_iterator = iter(train_loader)
            encodings = next(train_iterator, None)
        encodings = {k: v.cuda(non_blocking=True) for k, v in encodings.items()}

        # Self Loss
        encodings.update({
            'labels': encodings['input_ids'],
        })

        outputs = model(**encodings)

        loss = outputs.loss / accumulation
        loss.backward()
        step_loss += loss.item()

        del loss, outputs

        if (iteration + 1) % accumulation:
            continue

        clip_grad_norm_(model.parameters(), max_norm=config.train.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        writer.add_scalar("Autoencoder/loss", step_loss, global_step=step)

        if counts > 0 or counts == 0 and step_loss < config.train.threadhold:
            counts += 1
        if counts > config.train.lasting:
            break
        step_loss = 0

    model.save_pretrained("dump")

if __name__ == "__main__":
    main()