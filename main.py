from parser import parse_conllu, DependencyDataset, custom_collate_fn
from model import DependencyParserModel
from train_and_evaluate import train_and_evaluate, evaluate_model
def main():
    train_trees = parse_conllu('/content/sample_data/en_ewt-ud-train.conllu')
    dev_trees = parse_conllu('/content/sample_data/en_ewt-ud-dev.conllu')
    test_trees = parse_conllu('/content/sample_data/en_ewt-ud-test.conllu')

    word_vocab, label_vocab = build_vocab(train_trees + dev_trees + test_trees)
    train_dataset = DependencyDataset(train_trees, word_vocab, label_vocab)
    dev_dataset = DependencyDataset(dev_trees, word_vocab, label_vocab)
    test_dataset = DependencyDataset(test_trees, word_vocab, label_vocab)

    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True, collate_fn=custom_collate_fn)
    dev_loader = DataLoader(dev_dataset, batch_size=10, shuffle=False, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False, collate_fn=custom_collate_fn)

    model = DependencyParserModel(vocab_size=len(word_vocab), label_size=len(label_vocab), embedding_dim=100, hidden_dim=256)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion_head = nn.CrossEntropyLoss(ignore_index=-1)
    criterion_label = nn.CrossEntropyLoss(ignore_index=-1)

    train_and_evaluate(model, train_loader, optimizer, criterion_head, criterion_label, 1)

    dev_uas, dev_las = evaluate_model(model, dev_loader)
    print(f'Development Data - UAS: {dev_uas:.4f}, LAS: {dev_las:.4f}')

    test_uas, test_las = evaluate_model(model, test_loader)
    print(f'Test Data - UAS: {test_uas:.4f}, LAS: {test_las:.4f}')

if __name__ == '__main__':
    main()
