import os
from os.path import join as pjoin
import shutil, glob
import json
import random
import argparse
import re

from tqdm import tqdm

class Cleaner:
    def __init__(self):
        self.replacer= re.compile('\s')
        self.space_remover = re.compile(' +')

    def __call__(self, text):
        text = self.replacer.sub(' ', text)
        text = self.space_remover.sub(' ', text)
        return text

def preprocess_yelp(
    target_json: str='data/yelp/yelp_academic_dataset_review.json',
    preprocess_root: str='data/preprocessed/yelp',
    max_length: int=180,
    min_length: int=10,
    num_dev: int=5000,
    num_test: int=1000,
    seed: int=42
):
    positive = []
    negative = []
    cleaner = Cleaner()

    with open(target_json, 'r') as fd:
        for document in tqdm(fd, desc='Loading Yelp...'):
            sample = json.loads(document)
            if len(sample['text']) > max_length or len(sample['text']) < min_length or sample['stars'] == 3:
                continue
            
            if sum(map(str.isascii, sample['text'])) / len(sample['text']) < 0.5:
                continue

            text = cleaner(sample['text'])
            if sample['stars'] == 5:
                positive.append(text)
            elif sample['stars'] <= 2:
                negative.append(text)

    random.seed(seed)
    random.shuffle(positive)
    random.shuffle(negative)

    positive = positive[:min(len(negative), len(positive))]
    negative = negative[:min(len(negative), len(positive))]

    os.makedirs(preprocess_root, exist_ok=True)

    with open(pjoin(preprocess_root, 'sentences.test.1.txt'), 'w') as fd:
        fd.writelines(sentence + '\n' for sentence in positive[:num_test])
    with open(pjoin(preprocess_root, 'sentences.test.0.txt'), 'w') as fd:
        fd.writelines(sentence + '\n' for sentence in negative[:num_test])

    with open(pjoin(preprocess_root, 'sentences.dev.1.txt'), 'w') as fd:
        fd.writelines(sentence + '\n' for sentence in positive[num_test:num_test+num_dev])
    with open(pjoin(preprocess_root, 'sentences.dev.0.txt'), 'w') as fd:
        fd.writelines(sentence + '\n' for sentence in negative[num_test:num_test+num_dev])

    for split in ('dev', 'test'):
        for label in (0, 1):
            shutil.copyfile(pjoin(preprocess_root, f'sentences.{split}.{label}.txt'),
                            pjoin(preprocess_root, f'reference.{split}.{label}.0.txt'))

    with open(pjoin(preprocess_root, 'sentences.train.1.txt'), 'w') as fd:
        fd.writelines(sentence + '\n' for sentence in positive[num_test+num_dev:])
    with open(pjoin(preprocess_root, 'sentences.train.0.txt'), 'w') as fd:
        fd.writelines(sentence + '\n' for sentence in negative[num_test+num_dev:])

def preprocess_gyafc(
    target: str='data/gyafc',
    preprocess_root: str='data/preprocessed/gyafc',
):
    topics = {
        'em': 'Entertainment_Music',
        'fr': 'Family_Relationships'
    }

    for name, topic in topics.items():
        os.makedirs(f'{preprocess_root}_{name}', exist_ok=True)
        for split, raw_split_name in {'train': 'train', 'dev': 'tune', 'test': 'test'}.items():
            for i, style in enumerate(['formal', 'informal']):
                shutil.copyfile(f'{target}/{topic}/{raw_split_name}/{style}', f'{preprocess_root}_{name}/sentences.{split}.{i}.txt')
                if split == 'train':
                    continue

                for j in range(4):
                    shutil.copyfile(f'{target}/{topic}/{raw_split_name}/{style}.ref{j}', f'{preprocess_root}_{name}/reference.{split}.{1-i}.{j}.txt')
          
def preprocess_amazon(
    target: str='data/amazon',
    preprocess_root: str='data/preprocessed/amazon',
):  
    def preprocess(sentence):
        sentence = re.sub(r"\s+", " ", sentence)
        sentence = re.sub(r"(?<=[a-zA-Z]\s)(s|t|ve|m|re|d|ll)(?=\s)", r"'\1", sentence)
        sentence = re.sub(r" (?=[\.,'!?:;])", "", sentence)
        sentence = re.sub(r"(?<=\W[a-zA-z]\.)\s([a-zA-z]\.)", r"\1", sentence)
        sentence = re.sub(r"(?<=[,'!?:;])\.$", "", sentence)
        sentence = re.sub(r"(?<=\s)i(?=[\s|'])", "I", sentence)
        sentence = re.sub(r"(?:^|(?<=[\.!?]\s))([a-z])", lambda g: g.group(1).upper(), sentence)
        return sentence

    os.makedirs(preprocess_root, exist_ok=True)
    for file in glob.glob(pjoin(target, 'sentences.*')):
        file_name = file.rsplit('/', maxsplit=1)[-1]
        if file_name.startswith('sentences.test'):
            continue
        with open(file, 'r') as fi, open(f'{preprocess_root}/{file_name}.txt', 'w') as fo:
            for sentence in fi:
               fo.write(preprocess(sentence) + '\n')

    for label in (0, 1):
        shutil.copyfile(pjoin(preprocess_root, f'sentences.dev.{label}.txt'),
                        pjoin(preprocess_root, f'reference.dev.{label}.0.txt'))
    
    for file in glob.glob(pjoin(target, 'reference.*')):
        file_name = file.rsplit('/', maxsplit=1)[-1]
        label = file_name.rsplit('.', maxsplit=1)[-1]
        with open(file, 'r') as fi, \
             open(f'{preprocess_root}/sentences.test.{label}.txt', 'w') as fos, \
             open(f'{preprocess_root}/reference.test.{label}.0.txt', 'w') as fot:
             for sentence in fi:
                source, target = sentence.split('\t')
                fos.write(preprocess(source) + '\n')
                fot.write(preprocess(target) + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yelp', action='store_true')
    parser.add_argument('--gyafc', action='store_true')
    parser.add_argument('--amazon', action='store_true')
    parser.add_argument('--num_test', type=int, default=1000)
    parser.add_argument('--num_dev', type=int, default=5000)
    commandline = parser.parse_args()

    if commandline.yelp:
        preprocess_yelp(num_test=commandline.num_test, num_dev=commandline.num_dev)

    if commandline.gyafc:
        preprocess_gyafc()

    if commandline.amazon:
        preprocess_amazon()
            