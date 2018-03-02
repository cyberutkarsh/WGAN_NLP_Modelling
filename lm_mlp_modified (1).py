#!/usr/bin/env python
"""GAN Language Model."""
import numpy as np
import sys

import datetime
from data_utils import DataIterator, minibatch_to_1hot, minibatch_to_1hotnoise_fast, minibatch_to_1hotnoise, minibatch_to_1hotnoise_fast_softmax
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable, grad
import torch.optim as optim
import logging
import argparse
import os
from scipy.stats import entropy
from data_utils import test_bleu_5, test_bleu_7, valid_bleu_5, valid_bleu_7, CharDataIterator, TruncCharDataIterator, PTBGrammar
######################################################################################################
######################################################################################################
######################################################################################################
parser = argparse.ArgumentParser()

parser.add_argument(
    "--dataset",
    help="dataset to use",
    required=True
)
parser.add_argument(
    "--discriminator",
    help="conv/lstm",
    required=True
)
parser.add_argument(
    "--generator",
    help="mlp/peephole/lstm/bnmlp",
    required=True
)
parser.add_argument(
    "--objective",
    help="wgan/gan/lsgan",
    required=True
)
parser.add_argument(
    "--save_dir",
    help="save directory",
    required=True
)
parser.add_argument(
    "--level",
    help="word/char",
    required=True
)
args = parser.parse_args()
dataset = args.dataset
discriminator_type = args.discriminator
generator_type = args.generator
objective = args.objective
save_dir = args.save_dir
level = args.level
torch.set_num_threads(2)

######################################################################################################
######################################################################################################
######################################################################################################

now = datetime.datetime.now()
times = now.strftime("%Y-%m-%d %H:%M")
exp_name = 'time_%s__generator_%s__discriminator_%s__dataset_%s__objective_%s__level_%s' % (times, generator_type, discriminator_type, dataset, objective, level)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='log/%s' % (exp_name),
    filemode='w'
)

# define a new Handler to log to console as well
console = logging.StreamHandler()
# optional, set the logging level
console.setLevel(logging.INFO)
# set a format which is the same for console use
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
# tell the handler to use this format
console.setFormatter(formatter)
# add the handler to the root logger
logging.getLogger('').addHandler(console)

######################################################################################################
######################################################################################################
######################################################################################################

class LayerNormLSTM(nn.Module):
    r"""A long short-term memory (LSTM) cell with attention."""

    def __init__(self, input_size, hidden_size, batch_first=True):
        """Initialize params."""
        super(LayerNormLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = 1
        self.batch_first = batch_first

        self.input_weights = nn.Linear(input_size, 4 * hidden_size, bias=False)
        self.hidden_weights = nn.Linear(hidden_size, 4 * hidden_size, bias=False)
        self.bias = nn.Parameter(
            torch.zeros(4 * hidden_size,)
        )

        # Layer Normalization scale and shift params
        self.scale_add = 0.
        self.scale_mul = 1.
        self.b1 = nn.Parameter(
            torch.zeros(4 * hidden_size,)
        )
        self.b2 = nn.Parameter(
            torch.zeros(4 * hidden_size,)
        )
        self.b3 = nn.Parameter(
            torch.zeros(1 * hidden_size,)
        )

        self.s1 = nn.Parameter(
            torch.ones(4 * hidden_size,)
        )
        self.s2 = nn.Parameter(
            torch.ones(4 * hidden_size,)
        )
        self.s3 = nn.Parameter(
            torch.ones(1 * hidden_size,)
        )

    def _layer_norm(self, x, scale, shift):
        _eps = 1e-5
        x_mean = x.mean(1)
        x_var = ((x - x_mean.expand(x.size())) ** 2).sum(1)
        output = (x - x_mean.expand(x.size())) / torch.sqrt(
            (x_var.expand(x.size()) + _eps)
        )
        output = scale.unsqueeze(0).expand(output.size()) * \
            output + shift.unsqueeze(0).expand(output.size())

        return output

    def forward(self, input, hidden):
        """Propogate input through the network."""
        def recurrence(input, hidden):
            """Recurrence helper."""
            hx, cx = hidden  # n_b x hidden_dim
            hx = hx.squeeze()
            cx = cx.squeeze()
            gates = self._layer_norm(self.input_weights(input), self.b1, self.s1) + \
                self._layer_norm(self.hidden_weights(hx), self.b2, self.s2)
            gates = gates + self.bias.unsqueeze(0).expand(gates.size())
            ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

            ingate = F.sigmoid(ingate)
            forgetgate = F.sigmoid(forgetgate)
            cellgate = F.tanh(cellgate)
            outgate = F.sigmoid(outgate)

            cy = self._layer_norm(
                (forgetgate * cx) + (ingate * cellgate),
                self.b3,
                self.s3
            )

            hy = outgate * F.tanh(cy)  # n_b x hidden_dim
            return hy, cy

        if self.batch_first:
            input = input.transpose(0, 1)

        output = []
        steps = range(input.size(0))
        for i in steps:
            hidden = recurrence(input[i], hidden)
            output.append(hidden[0] if isinstance(hidden, tuple) else hidden)

        output = torch.cat(output, 0).view(input.size(0), *output[0].size())

        if self.batch_first:
            output = output.transpose(0, 1)

        return output, hidden
######################################################################################################
######################################################################################################
######################################################################################################

class Generator(nn.Module):
    """Generator."""

    def __init__(
        self,
        vocab_size,
        input_dim=100,
        hidden_dim=128,
        nlayers=1,
        bidirectional=False,
        dropout=0.,
        batch_first=True,
        layer_norm=False
    ):
        """Initialize model params."""
        super(Generator, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.nlayers = nlayers
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.batch_first = batch_first
        self.layer_norm = layer_norm
        self.num_directions = 2 if self.bidirectional else 1

        if self.layer_norm and self.nlayers > 1:
            raise NotImplementedError("Multi layer Layer Norm LSTM not implemented")

        if self.layer_norm and self.dropout > 0:
            raise NotImplementedError("Dropout Layer Norm LSTM not implemented")

        if self.layer_norm and self.bidirectional:
            raise NotImplementedError("Bi Layer Norm LSTM not implemented")

        if self.layer_norm and not self.batch_first:
            raise NotImplementedError("Seq first Layer Norm LSTM not implemented")

        if self.layer_norm:
            self.rnn = LayerNormLSTM(
                self.input_dim,
                self.hidden_dim,
            )

        else:
            self.rnn = nn.LSTM(
                self.input_dim,
                self.hidden_dim // 2 if self.bidirectional else self.hidden_dim,
                self.nlayers,
                bidirectional=self.bidirectional,
                batch_first=self.batch_first,
                dropout=self.dropout
            )

        self.decoder = nn.Linear(
            self.hidden_dim,
            self.vocab_size
        )

    def get_state(self, input):
        """Get cell states and hidden states."""
        batch_size = input.size(0) \
            if self.rnn.batch_first else input.size(1)
        h0_encoder = Variable(torch.zeros(
            self.rnn.num_layers * self.num_directions,
            batch_size,
            self.hidden_dim // 2 if self.bidirectional else self.hidden_dim,
        ))
        c0_encoder = Variable(torch.zeros(
            self.rnn.num_layers * self.num_directions,
            batch_size,
            self.hidden_dim // 2 if self.bidirectional else self.hidden_dim,
        ))

        return h0_encoder.cuda(), c0_encoder.cuda()

    def forward(self, input):
        """Propogate input through the network."""
        h0, c0 = self.get_state(input)
        h, (h_t, c_t) = self.rnn(input, (h0, c0))
        words = self.decoder(h.contiguous().view(-1, h.size(2)))
        words = F.softmax(words)
        return words.view(h.size(0), h.size(1), self.vocab_size)

######################################################################################################
######################################################################################################
######################################################################################################
class MLP(nn.Module):
    """Multi-layer perceptron."""

    def __init__(self, input_dim, hidden_dim, output_dim):
        """Initialize params."""
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, self.output_dim)

    def forward(self, x):
        """Propogate input through the network."""
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return F.softmax(self.fc4(x))
######################################################################################################
######################################################################################################
######################################################################################################

class BNMLP(nn.Module):
    """Multi-layer perceptron."""

    def __init__(self, input_dim, hidden_dim, output_dim):
        """Initialize params."""
        super(BNMLP, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, self.output_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)

    def forward(self, x):
        """Propogate input through the network."""
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        return F.softmax(self.fc4(x))
######################################################################################################
######################################################################################################
######################################################################################################

class MLPGenerator(nn.Module):
    """Generator."""

    def __init__(
        self,
        vocab_size,
        input_dim=100,
        hidden_dim=128,
        window=5,
        nlayers=1
    ):
        """Initialize model params."""
        super(MLPGenerator, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.nlayers = nlayers
        self.window = window

        self.mlp = MLP(self.window * self.input_dim, self.hidden_dim, self.vocab_size)

    def forward(self, input):
        """Propogate input through the network."""
        input = input.transpose(0, 1)
        padded_input = torch.cat(
            [Variable(torch.zeros(2, input.size(1), input.size(2)), requires_grad=False).cuda(), input],
            0
        )
        padded_input = torch.cat(
            [padded_input, Variable(torch.zeros(2, input.size(1), input.size(2)), requires_grad=False).cuda()],
            0
        )
        reps = []
        for i in range(input.size(0)):
            input_slice = padded_input[i:i + self.window]
            input_slice = input_slice.transpose(1, 2).contiguous().view(
                -1, input_slice.size(1)
            )
            rep = self.mlp(input_slice.t())
            reps.append(rep)
        reps = torch.stack(reps)
        return reps.transpose(0, 1)
######################################################################################################
######################################################################################################
######################################################################################################

class BNMLPGenerator(nn.Module):
    """Generator."""

    def __init__(
        self,
        vocab_size,
        input_dim=100,
        hidden_dim=128,
        window=5,
        nlayers=1
    ):
        """Initialize model params."""
        super(BNMLPGenerator, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.nlayers = nlayers
        self.window = window

        self.mlp = BNMLP(self.window * self.input_dim, self.hidden_dim, self.vocab_size)

    def forward(self, input):
        """Propogate input through the network."""
        input = input.transpose(0, 1)
        padded_input = torch.cat(
            [Variable(torch.zeros(2, input.size(1), input.size(2)), requires_grad=False).cuda(), input],
            0
        )
        padded_input = torch.cat(
            [padded_input, Variable(torch.zeros(2, input.size(1), input.size(2)), requires_grad=False).cuda()],
            0
        )
        reps = []
        for i in range(input.size(0)):
            input_slice = padded_input[i:i + self.window]
            input_slice = input_slice.transpose(1, 2).contiguous().view(
                -1, input_slice.size(1)
            )
            rep = self.mlp(input_slice.t())
            reps.append(rep)
        reps = torch.stack(reps)
        return reps.transpose(0, 1)

######################################################################################################
######################################################################################################
######################################################################################################
class SoftmaxPeepLSTM(nn.Module):
    """LSTM with peepholes at the softmax layer."""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim
    ):
        """Initialize params."""
        super(SoftmaxPeepLSTM, self).__init__()
        self.input_size = input_dim
        self.hidden_size = hidden_dim
        self.num_layers = 1
        self.batch_first = True

        self.input_weights = nn.Linear(input_dim, 4 * hidden_dim)
        self.hidden_weights = nn.Linear(hidden_dim, 4 * hidden_dim)
        self.output_peep_weights = nn.Linear(output_dim, 4 * hidden_dim)
        self.output_projection_weights = nn.Linear(hidden_dim, output_dim)

    def forward(self, input, hidden):
        """Propogate input through the network."""
        def recurrence(input, hidden):
            """Recurrence helper."""
            hx, cx, yx = hidden  # n_b x hidden_dim
            gates = self.input_weights(input) + \
                self.hidden_weights(hx) + self.output_peep_weights(yx)
            ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

            ingate = F.sigmoid(ingate)
            forgetgate = F.sigmoid(forgetgate)
            cellgate = F.tanh(cellgate)
            outgate = F.sigmoid(outgate)

            cy = (forgetgate * cx) + (ingate * cellgate)
            hy = outgate * F.tanh(cy)  # n_b x hidden_dim
            oy = F.softmax(self.output_projection_weights(hy))

            return hy, cy, oy

        if self.batch_first:
            input = input.transpose(0, 1)

        output = []
        output_peep = []
        steps = range(input.size(0))
        for i in steps:
            hidden = recurrence(input[i], hidden)
            output.append(hidden[0] if isinstance(hidden, tuple) else hidden)
            output_peep.append(hidden[2] if isinstance(hidden, tuple) else hidden)

        output = torch.cat(output, 0).view(input.size(0), *output[0].size())
        output_peep = torch.cat(output_peep, 0).view(
            input.size(0), *output_peep[0].size()
        )

        if self.batch_first:
            output = output.transpose(0, 1)
            output_peep = output_peep.transpose(0, 1)

        return output, output_peep, hidden
######################################################################################################
######################################################################################################
######################################################################################################

class SoftmaxPeepHoleGenerator(nn.Module):
    """Generator."""

    def __init__(
        self,
        vocab_size,
        input_dim=100,
        hidden_dim=128,
        nlayers=1,
        bidirectional=False,
        dropout=0.,
        batch_first=True,
        layer_norm=False
    ):
        """Initialize model params."""
        super(SoftmaxPeepHoleGenerator, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.nlayers = nlayers
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.batch_first = batch_first
        self.num_directions = 2 if self.bidirectional else 1

        self.rnn = SoftmaxPeepLSTM(
            self.input_dim,
            self.hidden_dim // 2 if self.bidirectional else self.hidden_dim,
            self.vocab_size
        )

    def get_state(self, input):
        """Get cell states and hidden states."""
        batch_size = input.size(0) \
            if self.rnn.batch_first else input.size(1)
        h0_encoder = Variable(torch.zeros(
            batch_size,
            self.hidden_dim // 2 if self.bidirectional else self.hidden_dim,
        ))
        c0_encoder = Variable(torch.zeros(
            batch_size,
            self.hidden_dim // 2 if self.bidirectional else self.hidden_dim,
        ))

        y0_encoder = Variable(torch.zeros(
            batch_size,
            self.vocab_size,
        ))

        return h0_encoder.cuda(), c0_encoder.cuda(), y0_encoder.cuda()

    def forward(self, input):
        """Propogate input through the network."""
        h0, c0, y0 = self.get_state(input)
        h, words, (h_t, c_t, y_t) = self.rnn(input, (h0, c0, y0))
        return words
######################################################################################################
######################################################################################################
######################################################################################################

class Discriminator(nn.Module):
    """Generator."""

    def __init__(
        self,
        input_dim,
        hidden_dim=128,
        nlayers=1,
        bidirectional=False,
        dropout=0.,
        batch_first=True,
        layer_norm=False
    ):
        """Initialize model params."""
        super(Discriminator, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.nlayers = nlayers
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.batch_first = batch_first
        self.layer_norm = layer_norm
        self.num_directions = 2 if self.bidirectional else 1

        if self.layer_norm and self.nlayers > 1:
            raise NotImplementedError("Multi layer Layer Norm LSTM not implemented")

        if self.layer_norm and self.dropout > 0:
            raise NotImplementedError("Dropout Layer Norm LSTM not implemented")

        if self.layer_norm and self.bidirectional:
            raise NotImplementedError("Bi Layer Norm LSTM not implemented")

        if self.layer_norm and not self.batch_first:
            raise NotImplementedError("Seq first Layer Norm LSTM not implemented")

        if self.layer_norm:
            self.rnn = LayerNormLSTM(
                self.input_dim,
                self.hidden_dim,
            )

        else:
            self.rnn = nn.LSTM(
                self.input_dim,
                self.hidden_dim // 2 if self.bidirectional else self.hidden_dim,
                self.nlayers,
                bidirectional=self.bidirectional,
                batch_first=self.batch_first,
                dropout=self.dropout
            )

        self.decoder = nn.Linear(
            self.hidden_dim,
            1
        )

    def get_state(self, input):
        """Get cell states and hidden states."""
        batch_size = input.size(0) \
            if self.rnn.batch_first else input.size(1)
        h0_encoder = Variable(torch.zeros(
            self.rnn.num_layers * self.num_directions,
            batch_size,
            self.hidden_dim // 2 if self.bidirectional else self.hidden_dim,
        ))
        c0_encoder = Variable(torch.zeros(
            self.rnn.num_layers * self.num_directions,
            batch_size,
            self.hidden_dim // 2 if self.bidirectional else self.hidden_dim,
        ))

        return h0_encoder.cuda(), c0_encoder.cuda()

    def forward(self, input):
        """Propogate input through the network."""
        h0, c0 = self.get_state(input)
        h, (h_t, c_t) = self.rnn(input, (h0, c0))

        if not self.layer_norm:
            if self.bidirectional:
                h_t = torch.cat((h_t[-1], h_t[-2]), 1)
            else:
                h_t = h_t[-1]

        return F.sigmoid(self.decoder(h_t))

######################################################################################################
######################################################################################################
######################################################################################################
class ReinforceDiscriminator(nn.Module):
    """Generator."""

    def __init__(
        self,
        input_dim,
        vocab_size,
        pad_token,
        hidden_dim=128,
        nlayers=1,
        bidirectional=False,
        dropout=0.,
        batch_first=True,
        layer_norm=False
    ):
        """Initialize model params."""
        super(ReinforceDiscriminator, self).__init__()
        self.input_dim = input_dim
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.pad_token = pad_token
        self.nlayers = nlayers
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.batch_first = batch_first
        self.layer_norm = layer_norm
        self.num_directions = 2 if self.bidirectional else 1

        if self.layer_norm and self.nlayers > 1:
            raise NotImplementedError("Multi layer Layer Norm LSTM not implemented")

        if self.layer_norm and self.dropout > 0:
            raise NotImplementedError("Dropout Layer Norm LSTM not implemented")

        if self.layer_norm and self.bidirectional:
            raise NotImplementedError("Bi Layer Norm LSTM not implemented")

        if self.layer_norm and not self.batch_first:
            raise NotImplementedError("Seq first Layer Norm LSTM not implemented")

        self.embedding = nn.Embedding(
            vocab_size,
            input_dim,
            self.pad_token
        )

        if self.layer_norm:
            self.rnn = LayerNormLSTM(
                self.input_dim,
                self.hidden_dim,
            )

        else:
            self.rnn = nn.LSTM(
                self.input_dim,
                self.hidden_dim // 2 if self.bidirectional else self.hidden_dim,
                self.nlayers,
                bidirectional=self.bidirectional,
                batch_first=self.batch_first,
                dropout=self.dropout
            )

        self.decoder = nn.Linear(
            self.hidden_dim,
            1
        )

    def get_state(self, input):
        """Get cell states and hidden states."""
        batch_size = input.size(0) \
            if self.rnn.batch_first else input.size(1)
        h0_encoder = Variable(torch.zeros(
            self.rnn.num_layers * self.num_directions,
            batch_size,
            self.hidden_dim // 2 if self.bidirectional else self.hidden_dim,
        ))
        c0_encoder = Variable(torch.zeros(
            self.rnn.num_layers * self.num_directions,
            batch_size,
            self.hidden_dim // 2 if self.bidirectional else self.hidden_dim,
        ))

        return h0_encoder.cuda(), c0_encoder.cuda()

    def forward(self, input):
        """Propogate input through the network."""
        h0, c0 = self.get_state(input)
        emb = self.embedding(input)
        h, (h_t, c_t) = self.rnn(emb, (h0, c0))

        if not self.layer_norm:
            if self.bidirectional:
                h_t = torch.cat((h_t[-1], h_t[-2]), 1)
            else:
                h_t = h_t[-1]

        return F.sigmoid(self.decoder(h_t))
######################################################################################################
######################################################################################################
######################################################################################################

class CharConvDiscriminator(nn.Module):
    """Generator."""

    def __init__(
        self,
        input_dim
    ):
        """Initialize model params."""
        super(CharConvDiscriminator, self).__init__()
        self.input_dim = input_dim

        self.conv1 = nn.Conv1d(
            input_dim, 64,
            2, stride=1, bias=False
        )
        self.pool1 = nn.MaxPool1d(2, stride=2)

        self.conv2 = nn.Conv1d(
            64, 32,
            2, stride=1, bias=False
        )
        self.pool2 = nn.MaxPool1d(2, stride=2)

        self.conv3 = nn.Conv1d(
            32, 1,
            2, stride=1, bias=False
        )
        self.pool3 = nn.MaxPool1d(2, stride=2)

    def forward(self, input):
        """Propogate input through the network."""
        conv1 = self.pool1(F.relu(self.conv1(input)))
        conv2 = self.pool2(F.relu(self.conv2(conv1)))
        conv3 = self.pool3(F.relu(self.conv3(conv2)))
        conv4 = self.conv4(conv3)

        return conv4
######################################################################################################
######################################################################################################
######################################################################################################

class WordConvDiscriminator(nn.Module):
    """Generator."""

    def __init__(
        self,
        input_dim
    ):
        """Initialize model params."""
        super(WordConvDiscriminator, self).__init__()
        self.input_dim = input_dim

        self.conv1 = nn.Conv1d(
            input_dim, 1000,
            2, stride=1, bias=False
        )
        self.bn1 = nn.BatchNorm1d(1000)
        self.pool1 = nn.MaxPool1d(2, stride=2)

        self.conv2 = nn.Conv1d(
            1000, 200,
            2, stride=1, bias=False
        )
        self.bn2 = nn.BatchNorm1d(200)
        self.pool2 = nn.MaxPool1d(2, stride=2)

        self.conv3 = nn.Conv1d(
            200, 1,
            2, stride=1, bias=False
        )
        self.pool3 = nn.MaxPool1d(2, stride=2)

    def forward(self, input):
        """Propogate input through the network."""
        input = input.transpose(1, 2)
        left_padding = (15 - input.size(2)) // 2
        right_padding = left_padding if input.size(2) % 2 else left_padding + 1
        padded_input = torch.cat([Variable(torch.zeros(input.size(0), input.size(1), left_padding), requires_grad=False).cuda(), input], 2)
        padded_input = torch.cat([padded_input, Variable(torch.zeros(input.size(0), input.size(1), right_padding), requires_grad=False).cuda()], 2)
        conv1 = self.pool1(F.relu(self.bn1(self.conv1(padded_input))))
        conv2 = self.pool2(F.relu(self.bn2(self.conv2(conv1))))
        conv3 = F.sigmoid(self.pool3(F.relu(self.conv3(conv2))))

        return conv3
######################################################################################################
######################################################################################################
######################################################################################################

def analyze_softmax_values(softmax_values):
    sequence_entropies = []
    for item in softmax_values:
        word_entropies = []
        for word in item:
            word_entropies.append(entropy(word))
        sequence_entropies.append(np.mean(word_entropies))

    return np.mean(sequence_entropies)

######################################################################################################
######################################################################################################
######################################################################################################
if __name__ == '__main__':

    if dataset == 'cmu':
        data_path = '/data/lisatmp4/subramas/datasets/cmu_hw_data/cmu-mthomework.train.en.unk'
    elif dataset == 'mehrad':
        data_path = "/home/ubuntu/merged_first"
##############################################################################
    if level == 'word':
        data_iterator = DataIterator(
            data_path=data_path,
            vocab_size=2000
        )
    else:
        data_iterator = CharDataIterator(
            data_path=data_path
        )
##############################################################################
    data_iterator.read_data()
    data_iterator.compute_vocab()
    total_sentences = len(data_iterator.sentences)
    logging.info('Found %d sentences in dataset ' % (total_sentences))
    logging.info('Mean sentence length : %.5f ' % (
        np.mean([len(sent) for sent in data_iterator.sentences])
    ))
    logging.info('Vocab size : %d ' % (len(data_iterator.word2id)))
    # Global params
    vocab_size = len(data_iterator.word2id)
    dim_word = 128
    rnn_hidden_dim_gen = 128
    rnn_hidden_dim_disc = 128
    num_epochs = 10
##############################################################################
    if generator_type == 'mlp':
        generator_1 = MLPGenerator(
            vocab_size,
            input_dim=dim_word,
            hidden_dim=rnn_hidden_dim_gen,
            window=5,
            nlayers=1
        ).cuda()
    elif generator_type == 'peephole':
        generator_1 = SoftmaxPeepHoleGenerator(
            vocab_size,
            input_dim=dim_word,
            hidden_dim=rnn_hidden_dim_gen,
        ).cuda()
    elif generator_type == 'lstm':
        generator_1 = Generator(
            vocab_size,
            input_dim=dim_word,
            hidden_dim=rnn_hidden_dim_gen,
        ).cuda()
    elif generator_type == 'bnmlp':
        generator_1 = BNMLPGenerator(
            vocab_size,
            input_dim=dim_word,
            hidden_dim=rnn_hidden_dim_gen,
            window=5,
            nlayers=1
        ).cuda()
    if discriminator_type == 'lstm':
        discriminator = Discriminator(
            vocab_size,
            hidden_dim=rnn_hidden_dim_disc,
            nlayers=1,
            bidirectional=False,
            dropout=0.,
            batch_first=True,
            layer_norm=False
        ).cuda()
    elif discriminator_type == 'cnn':
        discriminator = WordConvDiscriminator(
            vocab_size
        ).cuda()
##############################################################################
    optimizer_discriminator = optim.Adam(discriminator.parameters(), lr=2e-3, betas=(0.5, 0.999))

    optimizer_generator = optim.Adam(
        generator_1.parameters(),
        lr=2e-3,
        betas=(0.5, 0.999)
    )

    clamp_lower = -0.04
    clamp_upper = 0.04
    batch_size = 64
    num_batches = total_sentences / batch_size

##############################################################################

    start_length = 5
    for seq_length in range(start_length, 6):
        logging.info('Training with sequence length : %d ' % (seq_length))

        for i in range(num_epochs):
            minibatch_gen_losses = []
            minibatch_disc_losses = []
            for batch in range(int(num_batches)):

                ############################
                # (1) Update D network
                ###########################

                # Real examples
                for critic_iter in range(5):
                    minibatch = data_iterator.get_data_by_length(seq_length, 64)
                    minibatch = np.array([
                        [data_iterator.word2id[word] if word in data_iterator.word2id else data_iterator.word2id['<unk>'] for word in sentence]
                        for sentence in minibatch
                    ]).astype(np.int32)
                    minibatch = minibatch_to_1hot(minibatch, vocab_size)

                    # Noisy samples
                    minibatch_noise = Variable(torch.from_numpy(
                        np.random.randn(64, seq_length, dim_word).astype(np.float32)
                    )).cuda()

                    D1 = discriminator(minibatch)
                    fake = generator_1(minibatch_noise)
                    D2 = discriminator(fake)
                    if objective == 'wgan':
                        discriminator_loss = -0.5 * ((D1 - D2).mean())
                    elif objective == 'gan':
                        discriminator_loss = -0.5 * ((torch.log(D1) + torch.log(1 - D2)).mean())
                    elif objective == 'lsgan':
                        discriminator_loss = 0.5 * (((D1 - 1)**2).mean()) + 0.5 * (((D2) ** 2).mean())
                    elif objective == 'wgan_regularized':
                        
                        LAMBDA = 10
                        alpha = Variable(torch.from_numpy(
                        np.random.rand(64, 1, 1).astype(np.float32)
                    )).cuda()
                        
                        differences = fake - minibatch
                        interpolates = (minibatch + (alpha * differences)).cuda()
                        gradients = grad(discriminator(interpolates), interpolates, grad_outputs=torch.ones(discriminator(interpolates).size()).cuda(), create_graph=True, retain_graph=True, only_inputs=True)[0]
                        #slopes = torch.sqrt(torch.sum(torch.sum(gradients**2, dim=1), dim=1))
                        #gradient_penalty = torch.mean((slopes - 1.0)**2)
                        gradient_penalty = LAMBDA * ((gradients.norm(2, dim=1) - 1.0) ** 2).mean()
                 
                        discriminator_loss = -0.5 * ((D1 - D2).mean()) + gradient_penalty
                        
                   
                    optimizer_discriminator.zero_grad()
                    discriminator_loss.backward()
                    optimizer_discriminator.step()

                    if objective == 'wgan' or objective == 'gan':
                        # clamp parameters to a cube
                        for p in discriminator.parameters():
                            p.data.clamp_(clamp_lower, clamp_upper)

                ############################
                # (2) Update G network
                ############################

                # Noisy samples
                minibatch_noise = Variable(torch.from_numpy(
                    np.random.randn(64, seq_length, dim_word).astype(np.float32)
                )).cuda()
                D2 = discriminator(generator_1(minibatch_noise)).mean()
                if objective == 'wgan':
                    generator_loss = -.5 * D2.mean()
                elif objective == 'gan':
                    generator_loss = -.5 * torch.log(D2).mean()
                elif objective == 'lsgan':
                    generator_loss = .5 * ((D2 - 1)**2).mean()
                elif objective == 'wgan_regularized':
                    generator_loss = .5 * ((D2 - 1)**2).mean()
                optimizer_generator.zero_grad()
                generator_loss.backward()
                optimizer_generator.step()

                logging.info('[%d/%d/%d] Loss_D: %f Loss_G: %f' % (
                    i, batch, num_batches,
                    discriminator_loss.data[0], generator_loss.data[0]
                ))

                if batch % 50 == 0:
                    minibatch_noise = Variable(torch.from_numpy(
                        np.random.randn(64, seq_length, dim_word).astype(np.float32)
                    )).cuda()
                    generator_samples = generator_1(minibatch_noise).data.cpu().numpy()
                    # ent = analyze_softmax_values(generator_samples)
                    # logging.info(str(seq_length) + '\t' + str(i) + '\t' + str(batch) + '\t' + str(ent) + '\n')
                    # f_ent.write(str(seq_length) + '\t' + str(i) + '\t' + str(batch) + '\t' + str(ent) + '\n')
                    logging.info('Generating samples ... of sequence length : %d ' % (seq_length))
                    valid_sentences = 0
                    in_training_sentences = 0
                    for sentence in generator_samples:
                        argmax_words = np.argmax(sentence, axis=1)
                        sent_string = ' '.join([data_iterator.id2word[ind] for ind in argmax_words])
                        logging.info(sent_string)

                if generator_type == 'bnmlp':
                    generator_1.train()

            minibatch_noise = Variable(torch.from_numpy(
                np.random.randn(64, seq_length, dim_word).astype(np.float32)
            )).cuda()
            generator_samples = generator_1(minibatch_noise).data.cpu().numpy()
            for sentence in generator_samples:
                argmax_words = np.argmax(sentence, axis=1)
                sent_string = ' '.join([data_iterator.id2word[ind] for ind in argmax_words])
                logging.info(sent_string)
##############################################################################
        torch.save(
            generator_1.state_dict(),
            open(os.path.join(
                save_dir,
                exp_name + '__seqlen_%d.model.generator' % (seq_length)), 'wb')
        )
##############################################################################
        torch.save(
            discriminator.state_dict(),
            open(os.path.join(
                save_dir,
                exp_name + '__seqlen_%d.model.discriminator' % (seq_length)), 'wb')
        )
