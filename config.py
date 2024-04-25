import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, defaultdict
from torch.utils.data import DataLoader, Dataset

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
