import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
from itertools import count

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.optim import AdamW

from transformers import BertConfig

import hydra
from tqdm import tqdm

from style_bart import models, utils
from style_bart.data import datasets, utils as data_utils

@hydra.main("../../config", "pretrain_classifier", None)
def main(config):
    utils.set_seed(config.seed + 1)

    # Dataset
    train_dataset = datasets.StyledSentences(config.data, 'train')

    # Tokenizer and model
    tokenizer, _, style_token_ids = data_utils.get_styled_tokenizer(
        config.model.tokenizer, 
        config.model.get('additional_tokens', None)
    )

    if isinstance(config.model.arch, str):
        assert config.model.pretrained
        classifier = models.SoftBartForSentenceClassification.from_pretrained(
            config.model.arch,
            max_length=config.model.max_length,
            forced_bos_token_id=None,
            style_token_ids=style_token_ids
        )
            
        classifier.resize_token_embeddings(len(tokenizer))
        classifier.get_input_embeddings().weight.data[style_token_ids, :] \
            = classifier.get_input_embeddings().weight.data[[classifier.config.decoder_start_token_id], :]
    else:
        classifier = models.SoftBertForSentenceClassification(
            BertConfig(
                vocab_size=len(tokenizer),
                pad_token_id=tokenizer.pad_token_id,
                **config.model.arch
            ),
            style_token_ids=style_token_ids
        )

    classifier.cuda()

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
    if isinstance(config.model.arch, str):
        optimizer = AdamW([
            {'params': classifier.model.parameters()},
            {'params': classifier.classification_head.parameters(), 'lr': config.train.learning_rate_init}
        ], lr=config.train.learning_rate)
    else:
        optimizer = AdamW(classifier.parameters(), lr=config.train.learning_rate_init)

    # Tensorboard
    writer = SummaryWriter(log_dir='review')

    # Train
    classifier.train()
    train_iterator = iter(train_loader)

    cls_step = 0
    step_loss = 0
    counts = 0
    for iteration in tqdm(count(0), desc='Pratrain Classifier'):
        encodings = next(train_iterator, None)
        if encodings is None:
            train_iterator = iter(train_loader)
            encodings = next(train_iterator, None)
        sub_batch_size = len(encodings['input_ids'])

        encodings = {k: v.cuda(non_blocking=True) for k, v in encodings.items()}
        styles = encodings.pop('styles')

        encodings = {
            name: tensor.repeat(2, 1)
            for name, tensor in encodings.items()
        }
        encodings['styles'] = torch.cat([styles, 1 - styles])
        encodings['labels'] = torch.tensor([1, 0], device='cuda', dtype=torch.long).repeat_interleave(sub_batch_size)

        outputs = classifier(**encodings)
        loss = outputs.loss / accumulation
        loss.backward()
        step_loss += loss.item()

        del encodings, outputs, loss

        if (iteration + 1) % accumulation:
            continue
        clip_grad_norm_(classifier.parameters(), max_norm=config.train.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        cls_step += 1
        writer.add_scalar("Classifier/loss", step_loss, global_step=cls_step)

        if counts > 0 or counts == 0 and step_loss < config.train.threadhold:
            counts += 1
        if counts > config.train.lasting:
            break

        step_loss = 0

    classifier.save_pretrained("dump")

if __name__ == "__main__":
    main()