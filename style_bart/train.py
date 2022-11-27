import os, json
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
from os.path import join as pjoin
import subprocess

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.nn.functional import one_hot, pad
from torch.optim import AdamW

import hydra
from hydra.utils import to_absolute_path
from tqdm import tqdm

from style_bart import models, utils, transfer
from style_bart.data import datasets, utils as data_utils

@hydra.main("../config", "train_main", None)
def main(config):
    utils.set_seed(config.seed)
    utils.to_absolute_path_recursive(config)

    # Dataset
    train_dataset = datasets.StyledSentences(config.data, 'train')
    dev_sentences = [datasets.get_sentences(config.data, 'dev', label) for label in range(2)]

    # Tokenizer and model
    tokenizer, _, style_token_ids = data_utils.get_styled_tokenizer(
        config.model.tokenizer, 
        config.model.get('additional_tokens', None)
    )

    ## StyleBART
    model = models.StyleBart.from_pretrained(
        config.model.ae,
        max_length=config.model.max_length,
        tie_word_embeddings=True,
        forced_bos_token_id=None,
        num_beams=1,
        min_length=1,
        style_token_ids=style_token_ids,
    )

    model.cuda()

    ## Classifier
    with open(pjoin(config.model.cls, 'config.json')) as fd:
        cls_model_type = json.load(fd)["architectures"][0] 

    if cls_model_type == "SoftBartForSentenceClassification":
        classifier = models.SoftBartForSentenceClassification.from_pretrained(
            config.model.cls,
            max_length=config.model.max_length,
            style_token_ids=style_token_ids
        )
    elif cls_model_type == "SoftBertForSentenceClassification":
        classifier = models.SoftBertForSentenceClassification.from_pretrained(
            config.model.cls,
            max_length=config.model.max_length,
            style_token_ids=style_token_ids
        )

    classifier.cuda()

    ## Language Models
    lms = []
    for lm in config.model.lms:
        with open(pjoin(lm, 'config.json')) as fd:
            lm_model_type = json.load(fd)["architectures"][0]

        if lm_model_type == "BartForCausalLM":
            lm = models.BartForCausalLM.from_pretrained(
                lm,
                max_length=config.model.max_length,
                tie_word_embeddings=True,
            )
            lm.resize_token_embeddings(len(tokenizer))
            lm.get_input_embeddings().weight.data[style_token_ids, :] \
                = lm.get_input_embeddings().weight.data[[lm.config.decoder_start_token_id], :]
        elif lm_model_type == "GPT2LMHeadModel":
            lm = models.GPT2LMHeadModel.from_pretrained(
                lm, max_length=config.model.max_length,
            )
            lm.resize_token_embeddings(len(tokenizer))

        lm.cuda()
        lm.eval()
        lms.append(lm)

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
    train_cls_loader = data_utils.Loader(
        train_loader, 
        config.train.num_epochs \
            * (config.train.classifier_iteration / config.train.iteration)
    )
    train_loader = data_utils.Loader(train_loader, config.train.num_epochs)

    # Tensorboard & Output generator & Evaluator
    writer = SummaryWriter(log_dir='review')

    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config.train[
            'learning_rate' \
            if isinstance(config.model.arch, str) and config.model.pretrained is not False\
            else 'learning_rate_init'
        ])
    if cls_model_type == "SoftBartForSentenceClassification":
        cls_optimizer = AdamW([
            {'params': classifier.model.parameters()},
            {'params': classifier.classification_head.parameters(), 'lr': config.train.learning_rate_init}
        ], lr=config.train.learning_rate)
    else:
        cls_optimizer = AdamW(classifier.parameters(), lr=config.train.learning_rate_init)

    # Train
    progress_bar = tqdm(train_loader, desc="Train", unit_scale=1/accumulation, unit='step')
    train_iterator = iter(progress_bar)
    train_cls_iterator = iter(train_cls_loader)
    cls_step = 0
    step = 0
    while True:
        step_loss = 0
        for iteration in range(config.train.classifier_iteration * accumulation):
            encodings = next(train_cls_iterator, None)
            if encodings is None:
                break
            sub_batch_size = len(encodings['input_ids'])

            encodings = {k: v.cuda(non_blocking=True) for k, v in encodings.items()}
            styles = encodings.pop('styles')
            styles = torch.cat([styles, 1 - styles])

            encodings = {
                name: tensor.repeat(2, 1)
                for name, tensor in encodings.items()
            }
            encodings['styles'] = styles

            with torch.no_grad():
                model.eval()
                outputs = model.generate(
                    return_dict_in_generate=True,
                    forced_eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    bos_token_id=tokenizer.bos_token_id,
                    max_length=config.model.max_length+1,
                    **encodings
                )

            outputs = {
                'input_ids': outputs.sequences[:, 1:],
                'attention_mask': 
                    (~(outputs.sequences[:, 1:] == tokenizer.eos_token_id).cumsum(-1).bool() \
                        | (outputs.sequences[:, 1:] == tokenizer.eos_token_id)).long(),
                'distributions': outputs.distributions,
                'styles': styles,
                'labels': torch.tensor([1, 0], device='cuda', dtype=torch.long).repeat_interleave(sub_batch_size)
            }

            encodings['distributions'] = torch.full(list(encodings['input_ids'].shape) + [len(tokenizer)], \
                                                    -float('inf'), device=encodings['input_ids'].device)
            encodings['distributions'][one_hot(encodings['input_ids'], num_classes=len(tokenizer)).bool()] = 0
            encodings['labels'] = outputs['labels']

            for name in encodings:
                if encodings[name].ndim > 1:
                    if encodings[name].shape[1] < outputs[name].shape[1]:
                        encodings[name] = pad(
                            encodings[name],
                            (0, 0) * (encodings[name].dim() - 2) + (0, outputs[name].shape[1] - encodings[name].shape[1]),
                            value=(model.config.pad_token_id if name == 'input_ids' else 0)
                        )
                    elif encodings[name].shape[1] > outputs[name].shape[1]:
                        outputs[name] = pad(
                            outputs[name], 
                            (0, 0) * (outputs[name].dim() - 2) + (0, encodings[name].shape[1] - outputs[name].shape[1]),
                            value=(model.config.pad_token_id if name == 'input_ids' else 0)
                        )

                encodings[name] = torch.cat([encodings[name], outputs[name]], dim=0)

            classifier.train()
            outputs = classifier(**encodings)
            loss = outputs.loss / accumulation
            loss.backward()
            step_loss += loss.item()

            del encodings, outputs, loss

            if (iteration + 1) % accumulation:
                continue
            clip_grad_norm_(classifier.parameters(), max_norm=config.train.grad_clip)
            cls_optimizer.step()
            cls_optimizer.zero_grad(set_to_none=True)

            cls_step += 1
            writer.add_scalar("Train/Classifier Loss", step_loss, global_step=cls_step)
            step_loss = 0

        step_self_loss = 0
        step_cycle_loss = 0
        step_style_loss = 0
        step_fluency_loss = 0

        for iteration in range(config.train.iteration * accumulation):
            encodings = next(train_iterator, None)
            if encodings is None:
                break
            encodings = {k: v.cuda(non_blocking=True) for k, v in encodings.items()}
            styles = encodings.pop('styles')

            model.train()
            # Self Loss
            encodings.update({
                'labels': encodings['input_ids'],
                'styles': styles
            })

            outputs = model(**encodings)

            loss = outputs.loss / accumulation
            (loss * config.train.factor.self).backward()
            step_self_loss += loss.item()

            del loss, outputs

            # Cycle Loss
            reverse = dict(encodings)
            reverse['styles'] = 1 - styles
            del reverse['labels']

            generated = model.generate(
                return_dict_in_generate=True,
                forced_eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                bos_token_id=tokenizer.bos_token_id,
                max_length=config.model.max_length+1,
                **reverse
            )

            encodings.update({
                'attention_mask': 
                    (~(generated.sequences[:, 1:] == tokenizer.eos_token_id).cumsum(-1).bool() \
                        | (generated.sequences[:, 1:] == tokenizer.eos_token_id)).long(),
                'distributions': generated.distributions,
            })
            del encodings['input_ids']

            outputs = model(**encodings)
            cycle_loss = outputs.loss / accumulation
            step_cycle_loss += cycle_loss.item()

            del outputs

            # Style Loss
            encodings.update({
                'input_ids': generated.sequences[:, 1:],
                'styles': 1 - styles,
                'labels': torch.ones((encodings['distributions'].shape[0],), device=encodings['distributions'].device, dtype=torch.long)
            })
            
            classifier.eval()
            with utils.no_module_grad(classifier):
                outputs = classifier(**encodings)
            style_loss = outputs.loss / accumulation
            step_style_loss += style_loss.item()

            del generated, outputs
        
            # LM Loss
            del encodings['labels'], encodings['styles']
            
            fluency_loss = []
            for style, lm in enumerate(lms):
                index = styles == 1 - style
                if not index.any().item():
                    continue

                lm_encodings = {
                    key: value[index, ...]
                    for key, value in encodings.items()
                }
                distributions = lm_encodings.pop('distributions')

                with torch.no_grad():
                    outputs = lm(**lm_encodings)

                loss = torch.einsum('bsd,bsd->bs', distributions.softmax(-1), outputs.logits.float().log_softmax(-1))
                fluency_loss.append(loss[lm_encodings['attention_mask'].bool()])
            fluency_loss = -torch.cat(fluency_loss).mean() / accumulation
            step_fluency_loss += fluency_loss.item()

            (cycle_loss * config.train.factor.cycle
            + style_loss * config.train.factor.style
            + fluency_loss * config.train.factor.fluency).backward()

            del encodings, outputs, cycle_loss, style_loss, fluency_loss

            if (iteration + 1) % accumulation:
                continue

            clip_grad_norm_(model.parameters(), max_norm=config.train.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            step += 1
            writer.add_scalar("Train/Self Loss", step_self_loss, global_step=step)
            writer.add_scalar("Train/Cycle Loss", step_cycle_loss, global_step=step)
            writer.add_scalar("Train/Style Loss", step_style_loss, global_step=step)
            writer.add_scalar("Train/Fluency Loss", step_fluency_loss, global_step=step)
            loss = step_self_loss * config.train.factor.self \
                + step_cycle_loss * config.train.factor.cycle \
                + step_style_loss * config.train.factor.style \
                + step_fluency_loss * config.train.factor.fluency
            writer.add_scalar("Train/Loss", loss, global_step=step)
            step_self_loss = 0
            step_cycle_loss = 0
            step_style_loss = 0
            step_fluency_loss = 0

            if step % config.train.eval_per_step:
                continue

            paths = []
            for style, sentences in enumerate(dev_sentences):
                sentences = transfer.transfer(sentences, model, 1 - style, tokenizer)
                os.makedirs(f'out/{style}', exist_ok=True)
                paths.append(f'out/{style}/{step:07d}.txt')
                with open(paths[-1], 'w') as fd:
                    fd.writelines(sentence + '\n' for sentence in sentences)

            with utils.chdir_root("temp", *paths) as (temp_folder, *paths), \
                 utils.release_memory(model, classifier, *lms):
                proc = subprocess.Popen(["bash", 'evaluate/eval.sh', \
                                    "-d", config.data, "-s", "dev", "-t", temp_folder] + paths, \
                                        stdout=subprocess.PIPE)
                outs, _ = proc.communicate() 
            outputs = dict(zip(("Accuracy", "Similarity", "Joint(A,S)", "Fluency-CoLA", "Fluency-PPL"), \
                               map(float, outs.decode('utf-8').split(','))))
            for key, value in outputs.items():
                writer.add_scalar(f"Evaluation/{key}", value, global_step=step)

            if step % config.train.save_per_step:
                continue
            model.save_pretrained(f"dump/{step}")
            classifier.save_pretrained(f"dump/{step}_cls")
        else:
            continue
        break
    progress_bar.close()

    paths = []
    for style, sentences in enumerate(dev_sentences):
        sentences = transfer.transfer(sentences, model, 1 - style, tokenizer)
        os.makedirs(f'out/{style}', exist_ok=True)
        paths.append(f'out/{style}/{step:07d}.txt')
        with open(paths[-1], 'w') as fd:
            fd.writelines(sentence + '\n' for sentence in sentences)
    paths = list(map(to_absolute_path, paths))

    temp_folder = to_absolute_path("temp")
    with utils.chdir_root("temp", *paths) as (temp_folder, *paths), \
            utils.release_memory(model, classifier, *lms):
        proc = subprocess.Popen(["bash", 'evaluate/eval.sh', \
                            "-d", config.data, "-s", "dev", "-t", temp_folder] + paths, \
                                stdout=subprocess.PIPE)
        outs, _ = proc.communicate() 
    outputs = dict(zip(("Accuracy", "Similarity", "Joint(A,S)", "Fluency-CoLA", "Fluency-PPL"), \
                        map(float, outs.decode('utf-8').split(','))))
    for key, value in outputs.items():
        writer.add_scalar(f"Evaluation/{key}", value, global_step=step)

    model.save_pretrained(f"dump/{step}")
    classifier.save_pretrained(f"dump/{step}_cls")

if __name__ == "__main__":
    main()