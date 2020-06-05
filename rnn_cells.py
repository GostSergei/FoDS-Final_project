

import torch
import torch.nn as nn



class EncoderCell(nn.Module):
    name = 'EncoderCell'

    def __init__(self, vocab_size, hidden_size, rnn_type, scale=1, n_layers=1, batch_first=False):
        super().__init__()
        if batch_first:
            self.batch_dim = 0
            self.step_dim = 1
        else:
            self.batch_dim = 1
            self.step_dim = 0
        self.dim = 1

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.rnn_cell = rnn_type(hidden_size, hidden_size, scale, batch_first)

    def forward(self, input_, hidden):
        h_in_list = hidden.chunk(self.n_layers, self.dim)
        embedded = self.embedding(input_)
        x = embedded
        h_out_list = []
        for h_in in h_in_list:
            x = self.rnn_cell(x, h_in)
            h_out_list.append(x)
        hidden = torch.cat(h_out_list, self.dim)
        return hidden

    def init_hidden(self, batch_size=1):
        return torch.zeros(batch_size, self.hidden_size * self.n_layers, requires_grad=True)


class DecoderCell(nn.Module):
    name = 'DecoderCell'

    def __init__(self, vocab_size, hidden_size, rnn_type, scale=1, n_layers=1, batch_first=False):
        super().__init__()
        if batch_first:
            self.batch_dim = 0
            self.step_dim = 1
        else:
            self.batch_dim = 1
            self.step_dim = 0
        self.dim = 1

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.rnn_cell = rnn_type(hidden_size, hidden_size, scale, batch_first)
        self.h2o = nn.Linear(hidden_size, vocab_size)
        self.softmax = nn.LogSoftmax(dim=self.dim)

    def forward(self, input_, hidden):
        embedded = self.embedding(input_)
        embedded = nn.functional.relu(embedded)
        h_in_list = hidden.chunk(self.n_layers, self.dim)
        x = embedded
        h_out_list = []
        for h_in in h_in_list:
            x = self.rnn_cell(x, h_in)
            h_out_list.append(x)
        hidden = torch.cat(h_out_list, self.dim)
        h_last = x
        output = self.h2o(h_last)
        output = self.softmax(output)
        return output, hidden


# module RNN
class MyLinerRNN_cell(nn.Module):
    name = 'LinRNN'

    def __init__(self, input_size, hidden_size, scale=1, batch_first=False):
        super().__init__()

        if batch_first:
            self.batch_dim = 0
            self.step_dim = 1
        else:
            self.batch_dim = 1
            self.step_dim = 0
        self.dim = 1

        self.scale = scale
        if scale != 1:
            self.scaling(scale)

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), self.batch_dim)
        hidden = self.i2h(combined)
        return hidden

    def init_hidden(self, batch_size=1):
        return torch.zeros(batch_size, self.hidden_size, requires_grad=True)

    def scaling(self, scale):
        for param in self.parameters():
            param.data.mul_(scale)


class MyRNN_cell(nn.RNNCell):
    name = 'RNN'

    def __init__(self, input_size, hidden_size, scale=1, batch_first=False):
        super().__init__(input_size, hidden_size)

        if batch_first:
            self.batch_dim = 0
            self.step_dim = 1
        else:
            self.batch_dim = 1
            self.step_dim = 0
        self.dim = 1

        self.scale = scale
        if scale != 1:
            self.scaling(scale)

    def forward(self, input_, hidden):
        hidden = super().forward(input_, hidden)
        return hidden

    def reset_baises(self):
        self.bias_hh.data.fill_(0)
        self.bias_ih.data.fill_(0)
        self.h2o.bias.data.fill_(0)

    def init_hidden(self, batch_size=1):
        return torch.zeros(batch_size, self.hidden_size, requires_grad=True)

    def scaling(self, scale):
        for param in self.parameters():
            param.data.mul_(scale)


class MyLSTM_cell(nn.LSTMCell):
    name = 'LSTM'


    def __init__(self, input_size, hidden_size, scale=1, batch_first=False):
        super().__init__(input_size, hidden_size//2)
        self.n_chunks = 2
        if batch_first:
            self.batch_dim = 0
            self.step_dim = 1
        else:
            self.batch_dim = 1
            self.step_dim = 0
        self.dim = 1

        self.scale = scale
        if scale != 1:
            self.scaling(scale)

        self.init_baises(1)



    def forward(self, input_, hidden):
        h, s = hidden.chunk(self.n_chunks, self.dim)
        h, s = super().forward(input_, (h, s))
        hidden = torch.cat((h, s), self.dim)
        return hidden

    def init_baises(self, value):
        ii, if_, ig, io = self.bias_ih.chunk(4, 0)
        hi, hf, hg, ho = self.bias_hh.chunk(4, 0)
        hf.data.fill_(value / 2)
        if_.data.fill_(value / 2)

    def init_hidden(self, batch_size=1):
        h0 = torch.zeros(batch_size, self.hidden_size, requires_grad=True)
        s0 = torch.zeros(batch_size, self.hidden_size, requires_grad=True)
        hidden = torch.cat((h0, s0), self.dim)
        return hidden

    def scaling(self, scale):
        for param in self.parameters():
            param.data.mul_(scale)


class MyGRU_cell(nn.GRUCell):
    name = 'GRU'

    def __init__(self, input_size, hidden_size, scale=1, batch_first=False):
        super().__init__(input_size, hidden_size)
        self.scale = scale
        if scale != 1:
            self.scaling(scale)

    def forward(self, input_, hidden):
        hidden = super().forward(input_, hidden)
        return hidden

    def init_hidden(self, batch_size=1):
        return torch.zeros(batch_size, self.hidden_size, requires_grad=True)

    def scaling(self, scale):
        for param in self.parameters():
            param.data.mul_(scale)

