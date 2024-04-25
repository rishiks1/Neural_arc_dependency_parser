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
