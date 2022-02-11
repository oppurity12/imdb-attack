import torch
import torch.nn as nn
import math


class SSRGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(SSRGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.x2h = nn.Linear(input_size, 3 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 3 * hidden_size, bias=bias)
        self.reset_parameters()

        self.alpha_2_1 = nn.Parameter(torch.tensor(-1.0), requires_grad=False)
        self.beta_1_0 = nn.Parameter(torch.tensor(0.5), requires_grad=True)

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward_one_step(self, x, hidden):
        gate_x = self.x2h(x)
        gate_h = self.h2h(hidden)

        i_r, i_i, i_n = gate_x.chunk(3, 1)
        h_r, h_i, h_n = gate_h.chunk(3, 1)

        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + (resetgate * h_n))

        new_hidden = (1 - inputgate) * (newgate - hidden)

        return new_hidden

    def forward(self, x, hidden):
        m = nn.Sigmoid()
        alpha_2_0 = 1 - m(self.alpha_2_1)
        beta_2_0 = 1 - 1 / (2 * m(self.beta_1_0)) - \
            m(self.alpha_2_1) * m(self.beta_1_0)
        beta_2_1 = 1 / (2 * m(self.beta_1_0))

        half_step_hidden = hidden + \
            m(self.beta_1_0) * self.forward_one_step(x, hidden)
        new_hidden = alpha_2_0 * hidden + beta_2_0 * self.forward_one_step(x, hidden) + \
            m(self.alpha_2_1) * half_step_hidden + beta_2_1 * \
            self.forward_one_step(x, half_step_hidden)

        return new_hidden


class SSRGRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, batch_first=False):
        super(SSRGRUModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.batch_first = batch_first
        self.rnn = []
        for i in range(layer_dim):
            if i == 0:
                self.rnn.append(SSRGRUCell(input_dim, hidden_dim))
                continue
            self.rnn.append(SSRGRUCell(hidden_dim, hidden_dim))

        self.rnn = nn.ModuleList(self.rnn)

    def forward(self, x):
        # x shape = (seq_length, batch_size, input_dim)
        if self.batch_first:
            # x shape =  (batch_size, seq_length, input_dim) -> (seq_length, batch_size, input_dim)
            x = x.permute(1, 0, 2)

        h0 = torch.zeros(self.layer_dim, x.size(
            1), self.hidden_dim).to(x.device)
        for idx, layer in enumerate(self.rnn):
            hn = h0[idx, :, :]
            outs = []

            for seq in range(x.size(0)):
                hn = layer(x[seq, :, :], hn)
                # print(hn.shape)
                outs.append(hn)

            x = torch.stack(outs, 0)

        if self.batch_first:
            x = x.permute(1, 0, 2)

        return x, x[-1]
