import torch
from collections import deque, defaultdict
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

class DependencyTree:
    def __init__(self, tokens):
        self.tokens = tokens
        self.size = len(tokens)
        self.heads = [-1] * self.size  # Initialize heads with -1
        self.labels = [None] * self.size

class Configuration:
    def __init__(self, tree):
        self.tree = tree
        self.stack = [0]  # Root token is always at index 0
        self.buffer = deque(range(1, tree.size))

    def shift(self):
        if self.buffer:
            self.stack.append(self.buffer.popleft())

    def left_arc(self):
        if len(self.stack) > 1:
            dep = self.stack.pop(-2)
            self.tree.heads[dep] = self.stack[-1]

    def right_arc(self):
        if len(self.stack) > 1:
            dep = self.stack.pop()
            self.tree.heads[dep] = self.stack[-1]
            
def parse_conllu(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = file.read().strip().split('\n\n')
    trees = []
    for sentence in data:
        tokens = []
        for line in sentence.split('\n'):
            if line.startswith('#') or '-' in line.split('\t')[0] or '.' in line.split('\t')[0]:
                continue  # Skip comment lines and non-integer token IDs
            parts = line.split('\t')
            if len(parts) == 10:
                tokens.append({
                    'id': int(parts[0]),
                    'form': parts[1],
                    'lemma': parts[2],
                    'upos': parts[3],
                    'xpos': parts[4],
                    'feats': parts[5],
                    'head': int(parts[6]),
                    'deprel': parts[7],
                    'deps': parts[8],
                    'misc': parts[9]
                })
        trees.append(DependencyTree(tokens))
    return trees


def build_vocab(trees):
    word_vocab = defaultdict(lambda: len(word_vocab))
    label_vocab = defaultdict(lambda: len(label_vocab))
    for tree in trees:
        for token in tree.tokens:
            _ = word_vocab[token['form'].lower()]
            _ = label_vocab[token['deprel']]
    return dict(word_vocab), dict(label_vocab)


class DependencyDataset(Dataset):
    def __init__(self, trees, word_vocab, label_vocab):
        self.trees = trees
        self.word_vocab = word_vocab
        self.label_vocab = label_vocab

    def __len__(self):
        return len(self.trees)

    def __getitem__(self, idx):
        tree = self.trees[idx]
        words = [self.word_vocab[token['form'].lower()] for token in tree.tokens]
        heads = [token['head'] for token in tree.tokens]
        labels = [self.label_vocab[token['deprel']] for token in tree.tokens]
        return torch.tensor(words, dtype=torch.long), torch.tensor(heads, dtype=torch.long), torch.tensor(labels, dtype=torch.long)

def custom_collate_fn(batch):
    words, heads, labels = zip(*batch)
    words_padded = pad_sequence([seq.clone().detach() for seq in words], batch_first=True, padding_value=0)
    heads_padded = pad_sequence([seq.clone().detach() for seq in heads], batch_first=True, padding_value=-1)  # Use an ignore index
    labels_padded = pad_sequence([seq.clone().detach() for seq in labels], batch_first=True, padding_value=-1)  # Use an ignore index
    return words_padded, heads_padded, labels_padded


