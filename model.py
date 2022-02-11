import torch
import torch.nn as nn
import torch.optim as optim

import pytorch_lightning as pl

from rnn_models import gru_alpha


class IMDBModel(nn.Module):
    def __init__(
            self,
            rnn_model, hidden_dim,
            n_vocab, embed_dim=128, num_layers=1,
            num_classes=2, lr=1e-3, alpha=1) -> None:

        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.lr = lr
        self.embed = nn.Embedding(n_vocab, embed_dim)
        if rnn_model is gru_alpha:
            self.rnn = rnn_model(embed_dim, hidden_dim,
                                 num_layers, alpha, batch_first=True)
        else:
            self.rnn = rnn_model(
                embed_dim, hidden_dim,
                num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def get_embed_vector(self, x):
        return self.embed(x)

    def forward(self, x):
        x = self.embed(x)
        x, _ = self.rnn(x)
        h_t = x[:, -1, :]
        logit = self.fc(h_t)
        return logit

    def forward2(self, x):
        x, _ = self.rnn(x)
        h_t = x[:, -1, :]
        logit = self.fc(h_t)
        return logit
