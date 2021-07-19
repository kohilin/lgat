import torch
import torch.nn as nn

from transformers import AutoTokenizer

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class RNNEncoder(nn.Module):
    def __init__(self, tokenizer='bert-base-uncased', hidden_size=128, max_length=50, rnn='gru'):
        super(RNNEncoder, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.max_length = max_length
        self.hidden_size = hidden_size
        self.emb = nn.Embedding(self.tokenizer.vocab_size, hidden_size)
        RNN = nn.LSTM if rnn == 'lstm' else nn.GRU
        self.rnn = RNN(input_size=self.hidden_size, hidden_size=self.hidden_size, bidirectional=True, batch_first=True)

        self.to(DEVICE)

    def forward(self, obs):
        inputs = self.tokenizer(obs, return_tensors='pt', padding=True, max_length=self.max_length, truncation=True)
        input_ids = inputs['input_ids'].to(DEVICE)
        embs = self.emb(input_ids)

        hiddens, _ = self.rnn(embs)

        return hiddens[:, -1, :]

    @property
    def size(self):
        return self.hidden_size * 2

