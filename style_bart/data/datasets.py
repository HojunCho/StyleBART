import glob
from hydra.utils import to_absolute_path

def get_sentences(name: str, split: str, label: int, root: str='data/preprocessed'):
    with open(to_absolute_path(f"{root}/{name}/sentences.{split}.{label}.txt")) as fd:
        dataset = [sentence.strip() for sentence in fd.readlines()]

    return dataset

def get_references(name: str, split: str, label: int, root: str='data/preprocessed'):
    references = []
    for path in glob.glob(to_absolute_path(f"{root}/{name}/reference.{split}.{label}.?.txt")):
        with open(path) as fd:
            references.append(list(map(str.strip, fd)))

    return references

class StyledSentences:
    def __init__(self, name: str, split: str, root: str='data/preprocessed'):
        paths = glob.glob(to_absolute_path(f'{root}/{name}/sentences.{split}.*.txt'))
        datasets = [None] * len(paths)

        for file_path in paths:
            label = int(file_path.rsplit('.', maxsplit=2)[1])
            datasets[label] = get_sentences(name, split, label, root)
        
        self.datasets = datasets

    def __len__(self):
        return max(map(len, self.datasets)) * len(self.datasets)

    def __getitem__(self, index):
        if len(self) <= index:
            raise IndexError()

        label = index % len(self.datasets)
        sentence_id = index // len(self.datasets)

        return self.datasets[label][sentence_id % len(self.datasets[label])], label
