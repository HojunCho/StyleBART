import sys

import torch
from torch.utils.data import DataLoader, IterableDataset

import click
from tqdm import tqdm

from . import models
from .data import utils as data_utils

class SimpleIterableWrapper(IterableDataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
    
    def __iter__(self):
        for sample in self.dataset:
            yield sample

def transfer(sentences, model, label, tokenizer, batch_size=64):
    loader = DataLoader(sentences, batch_size=batch_size, collate_fn=data_utils.SentenceCollator(tokenizer, model.config.max_length))
    
    sentences = []
    for encodings in tqdm(loader):
        encodings['styles'] = torch.full((len(encodings['input_ids']), ), label, dtype=torch.long)
        encodings = {k: v.cuda() for k, v in encodings.items()}

        outputs = model.generate(
            return_dict_in_generate=True,
            no_repeat_ngram_size=3,
            forced_eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            decoder_hard=True,
            **encodings
        )

        for sequence in outputs.sequences:
            sentences.append(tokenizer.decode(sequence[1:], skip_special_tokens=True))

    return sentences

@click.command()
@click.option('-m', '--model', required=True, type=click.Path(exists=True, file_okay=False), help="Model path to inference")
@click.option('-l', '--label', required=True, type=int, help="Transfer target label")
@click.option('-t', '--tokenizer', default='facebook/bart-base', type=str, help="Tokenizer path", show_default=True)
@click.option('--batch_size', default=64, type=int, help="Batch size", show_default=True)
@click.argument('prompt', default='')
def main(model, label, tokenizer, batch_size, prompt):
    tokenizer, _, style_token_ids = data_utils.get_styled_tokenizer(tokenizer)

    model = models.StyleBart.from_pretrained(model, style_token_ids=style_token_ids)
    model.cuda()

    if prompt:
        prompts = [prompt]
    else:
        prompts = SimpleIterableWrapper(line.strip() for line in sys.stdin)
    
    for sentence in transfer(prompts, model, label, tokenizer, batch_size):
        print(sentence)

if __name__ == "__main__":
    main()
