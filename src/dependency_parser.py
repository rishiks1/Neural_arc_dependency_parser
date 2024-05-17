import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils import parse_conllu, build_vocab, DependencyDataset, custom_collate_fn

class DependencyParserModel(nn.Module):
    def __init__(self, vocab_size, label_size, embedding_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=1, bidirectional=True)
        self.head_predictor = nn.Linear(hidden_dim * 2, vocab_size)  # Adjust output dimensions if needed
        self.label_predictor = nn.Linear(hidden_dim * 2, label_size)

    def forward(self, words):
        embeds = self.embed(words)
        lstm_out, _ = self.lstm(embeds)
        heads = self.head_predictor(lstm_out).squeeze()
        labels = self.label_predictor(lstm_out)
        return heads, labels  # Directly return logits without applying softmax

def train_and_evaluate(model, data_loader, optimizer, criterion_head, criterion_label, epoch_count):
    for epoch in range(epoch_count):
        model.train()
        total_loss = 0
        for words, true_heads, true_labels in data_loader:
            pred_heads, pred_labels = model(words)

            loss_heads = criterion_head(pred_heads.transpose(1, 2), true_heads)
            loss_labels = criterion_label(pred_labels.transpose(1, 2), true_labels)

            loss = loss_heads + loss_labels
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f'Epoch {epoch+1}: Loss {total_loss / len(data_loader)}')

def evaluate_model(model, data_loader):
    model.eval()
    total_correct_heads = 0
    total_correct_labels = 0
    total_tokens = 0

    with torch.no_grad():
        for words, true_heads, true_labels in data_loader:
            pred_heads, pred_labels = model(words)

            # Assuming the output dimensions need adjusting:
            pred_heads = pred_heads.argmax(-1)
            pred_labels = pred_labels.argmax(-1)

            correct_heads = (pred_heads == true_heads) & (true_heads != -1)  # Ignore padding
            correct_labels = (pred_labels == true_labels) & (true_labels != -1) & (pred_heads == true_heads)  # Correct label at correct head

            total_correct_heads += correct_heads.sum().item()
            total_correct_labels += correct_labels.sum().item()
            total_tokens += (true_heads != -1).sum().item()  # Count non-padded tokens

    uas = total_correct_heads / total_tokens
    las = total_correct_labels / total_tokens
    return uas, las

def train_and_evaluate_language(language, model, optimizer, criterion_head, criterion_label, epoch_count=1, batch_size=10):
    train_trees = parse_conllu(f'data/{language}-ud-train.conllu')
    dev_trees = parse_conllu(f'data/{language}-ud-dev.conllu')
    test_trees = parse_conllu(f'data/{language}-ud-test.conllu')

    word_vocab, label_vocab = build_vocab(train_trees + dev_trees + test_trees)
    train_dataset = DependencyDataset(train_trees, word_vocab, label_vocab)
    dev_dataset = DependencyDataset(dev_trees, word_vocab, label_vocab)
    test_dataset = DependencyDataset(test_trees, word_vocab, label_vocab)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)

    train_and_evaluate(model, train_loader, optimizer, criterion_head, criterion_label, epoch_count)

    dev_uas, dev_las = evaluate_model(model, dev_loader)
    print(f'{language.capitalize()} Development Data - UAS: {dev_uas:.4f}, LAS: {dev_las:.4f}')

    test_uas, test_las = evaluate_model(model, test_loader)
    print(f'{language.capitalize()} Test Data - UAS: {test_uas:.4f}, LAS: {test_las:.4f}')

def main():
    languages = ['en_ewt', 'es_ancora']
    for language in languages:
        print(f'Training and evaluating on {language} dataset...')

        train_trees = parse_conllu(f'data/{language}-ud-train.conllu')
        dev_trees = parse_conllu(f'data/{language}-ud-dev.conllu')
        test_trees = parse_conllu(f'data/{language}-ud-test.conllu')

        word_vocab, label_vocab = build_vocab(train_trees + dev_trees + test_trees)

        model = DependencyParserModel(vocab_size=len(word_vocab), label_size=len(label_vocab), embedding_dim=100, hidden_dim=256)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion_head = nn.CrossEntropyLoss(ignore_index=-1)
        criterion_label = nn.CrossEntropyLoss(ignore_index=-1)

        train_and_evaluate_language(language, model, optimizer, criterion_head, criterion_label, epoch_count=1)

if __name__ == '__main__':
    main()
