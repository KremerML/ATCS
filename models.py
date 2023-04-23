import torch
import torch.nn as nn
from nltk.tokenize import word_tokenize
import numpy as np
import time 

class NLIClassifier(nn.Module):
    def __init__(self, config):
        super(NLIClassifier, self).__init__()

        self.fc_dim = config['fc_dim']
        self.n_classes = config['n_classes']
        self.enc_lstm_dim = config['enc_lstm_dim']
        self.encoder_type = config['encoder_type']
        self.word_emb_dim = config['word_emb_dim']

        self.encoder = eval(self.encoder_type)(config)
        
        self.inputdim = 0
        if self.encoder_type == "BasicEncoder":
            self.inputdim = self.word_emb_dim * 4
        elif self.encoder_type == "LSTMEncoder":
            self.inputdim = 4*self.enc_lstm_dim
        elif self.encoder_type == "biLSTMEncoder" or self.encoder_type == "biLSTMMaxPoolEncoder":
            self.inputdim = 4*2*self.enc_lstm_dim
 
        self.classifier = nn.Sequential(
            nn.Linear(self.inputdim, self.fc_dim),
            nn.Linear(self.fc_dim, self.fc_dim),
            nn.Linear(self.fc_dim, self.n_classes)
            )
    def forward(self, s1, s2):
        # s1 : (s1, s1_len)
        u = self.encoder(s1)
        v = self.encoder(s2)

        features = torch.cat((u, v, torch.abs(u-v), u*v), 1)
        output = self.classifier(features)
        return output

    def encode(self, s1):
        emb = self.encoder(s1)
        return emb        


# Basic encoder. Computes the average of GLoVe word embeddings as the sentence representation.
class BasicEncoder(nn.Module):
    def __init__(self, config):
        super(BasicEncoder, self).__init__()
        self.word_emb_dim = config['word_emb_dim']

    def forward(self, emb):
        input_batch, _ = emb  # Unpack the input_batch and ignore the lengths tensor

        emb = torch.mean(input_batch, dim=1)

        return emb
        
# Simple unidirectional LSTM encoder. The final hidden state of the LSTM is used as the sentence representation
class LSTMEncoder(nn.Module):
    def __init__(self, config):
        super(LSTMEncoder, self).__init__()
        self.bsize = config['bsize']
        self.word_emb_dim = config['word_emb_dim']
        self.enc_lstm_dim = config['enc_lstm_dim']
        self.dpout_model = config['dpout_model']

        self.enc_lstm = nn.LSTM(self.word_emb_dim, self.enc_lstm_dim, num_layers=1,
                                bidirectional=False, dropout=self.dpout_model)

    def forward(self, sent_tuple):
        # sent_len [max_len, ..., min_len] (batch)
        # sent (seqlen x batch x worddim)
        sent, sent_len = sent_tuple

        # Handling padding in Recurrent Networks
        sent_packed = nn.utils.rnn.pack_padded_sequence(sent, sent_len, batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.enc_lstm(sent_packed)

        sent_output = h_n[0, ...]
        
        return sent_output

# Simple bidirectional LSTM (BiLSTM)
# last hidden state of forward and backward layers are concatenated as the sentence representation
class biLSTMEncoder(nn.Module):
    def __init__(self, config):
        super(biLSTMEncoder, self).__init__()
        self.word_emb_dim = config['word_emb_dim']
        self.enc_lstm_dim = config['enc_lstm_dim']
        self.dpout_model = config['dpout_model']

        self.enc_lstm = nn.LSTM(self.word_emb_dim, self.enc_lstm_dim, 
                                num_layers=1, bidirectional=True, dropout=self.dpout_model)
        
    def forward(self, sent_tuple):
        sent, sent_len = sent_tuple

        
        # Handling padding in Recurrent Networks
        sent_packed = nn.utils.rnn.pack_padded_sequence(sent, sent_len, batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.enc_lstm(sent_packed)

        # Concatenate the last hidden states of the forward and backward LSTM layers
        emb = torch.cat((h_n[0], h_n[1]), dim=-1)

        return emb

# BiLSTM with max pooling applied to the concatenation of word-level hidden states from
# both directions to retrieve sentence representations
class biLSTMMaxPoolEncoder(nn.Module):
    def __init__(self, config):
        super(biLSTMMaxPoolEncoder, self).__init__()
        self.word_emb_dim = config['word_emb_dim']
        self.enc_lstm_dim = config['enc_lstm_dim']
        self.dpout_model = config['dpout_model']

        self.enc_lstm = nn.LSTM(self.word_emb_dim, self.enc_lstm_dim, 
                                num_layers=1, bidirectional=True, dropout=self.dpout_model, batch_first=True)

    def forward(self, sent_tuple):

        sent, sent_len = sent_tuple

        # Handling padding in Recurrent Networks
        sent_packed = nn.utils.rnn.pack_padded_sequence(sent, sent_len, batch_first=True, enforce_sorted=False)
        sent_output, _ = self.enc_lstm(sent_packed)
        sent_output, _ = nn.utils.rnn.pad_packed_sequence(sent_output, batch_first=True)
        
        # Max Pooling
        emb, _ = torch.max(sent_output, dim=1)

        return emb