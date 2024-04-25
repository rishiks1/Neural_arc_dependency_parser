import torch
from torch.utils.data import DataLoader
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


