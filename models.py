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

        self.encoder = eval(self.encoder_type)(config)
        
        self.inputdim = 4*2*self.enc_lstm_dim
        self.inputdim = self.inputdim/2 if self.encoder_type == "LSTMEncoder" \
                                        else self.inputdim
        self.inputdim = 300 if self.encoder_type == "BasicEncoder" else self.inputdim
        
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


# averaging GLoVe word embeddings to obtain sentence representations
class BasicEncoder(nn.Module):
    def __init__(self, config):
        super(BasicEncoder, self).__init__()
        self.vector_embeddings = config['vector_embeddings']
        self.vocab_size = len(self.vector_embeddings)
        self.word_emb_dim = config['word_emb_dim']

        self.embedding = nn.Embedding(self.vocab_size, self.word_emb_dim)
        
        # Convert the dictionary to a tensor before copying
        vector_embeddings_tensor = torch.FloatTensor(np.array(list(self.vector_embeddings.values())))
        print("Shape of vector_embeddings_tensor:", vector_embeddings_tensor.shape)
        self.embedding.weight.data.copy_(vector_embeddings_tensor)


    def forward(self, emb):
        input_batch, _ = emb  # Unpack the input_batch and ignore the lengths tensor
        input_batch = input_batch.long()
        print("Max index in input_batch:", input_batch.max())
        print("Min index in input_batch:", input_batch.min())
        print("Vocab size:", self.vocab_size)

        emb = self.embedding(input_batch)
        emb = torch.mean(emb, dim=1)

        return emb
        
class LSTMEncoder(nn.Module):
    def __init__(self, config):
        super(LSTMEncoder, self).__init__()
        self.bsize = config['bsize']
        self.word_emb_dim = config['word_emb_dim']
        self.enc_lstm_dim = config['enc_lstm_dim']
        self.dpout_model = config['dpout_model']

        self.enc_lstm = nn.LSTM(self.word_emb_dim, self.enc_lstm_dim, 1,
                                bidirectional=False, dropout=self.dpout_model)

    def forward(self, sent_tuple):
        # sent_len [max_len, ..., min_len] (batch)
        # sent (seqlen x batch x worddim)

        sent, sent_len = sent_tuple

        # Sort by length (keep idx)
        sent_len, idx_sort = np.sort(sent_len)[::-1], np.argsort(-sent_len)
        sent = sent.index_select(1, torch.cuda.LongTensor(idx_sort))

        # Handling padding in Recurrent Networks
        sent_packed = nn.utils.rnn.pack_padded_sequence(sent, sent_len)
        sent_output = self.enc_lstm(sent_packed)[1][0].squeeze(0)  # batch x 2*nhid

        # Un-sort by length
        idx_unsort = np.argsort(idx_sort)
        emb = sent_output.index_select(0, torch.cuda.LongTensor(idx_unsort))

        return emb

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

        # Sort by length (keep idx)
        sent_len, idx_sort = np.sort(sent_len)[::-1], np.argsort(-sent_len)
        sent = sent.index_select(1, torch.cuda.LongTensor(idx_sort))

        # Handling padding in Recurrent Networks
        sent_packed = nn.utils.rnn.pack_padded_sequence(sent, sent_len)
        _, (h_n, _) = self.enc_lstm(sent_packed)

        # Un-sort by length
        idx_unsort = np.argsort(idx_sort)
        h_n = h_n.index_select(1, torch.cuda.LongTensor(idx_unsort))

        # Concatenate the last hidden states of the forward and backward LSTM layers
        emb = torch.cat((h_n[0], h_n[1]), dim=-1)

        return emb


class BiLSTMMaxPoolEncoder(nn.Module):
    def __init__(self, config):
        super(BiLSTMMaxPoolEncoder, self).__init__()
        self.word_emb_dim = config['word_emb_dim']
        self.enc_lstm_dim = config['enc_lstm_dim']
        self.dpout_model = config['dpout_model']

        self.enc_lstm = nn.LSTM(self.word_emb_dim, self.enc_lstm_dim, 
                                num_layers=1, bidirectional=True, dropout=self.dpout_model)
        self.proj_enc = nn.Linear(2 * self.enc_lstm_dim, 
                                  2 * self.enc_lstm_dim, bias=False)

    def forward(self, sent_tuple):
        sent, sent_len = sent_tuple
        bsize = sent.size(1)

        # Sort by length (keep idx)
        sent_len, idx_sort = np.sort(sent_len)[::-1], np.argsort(-sent_len)
        sent = sent.index_select(1, torch.cuda.LongTensor(idx_sort))

        # Handling padding in Recurrent Networks
        sent_packed = nn.utils.rnn.pack_padded_sequence(sent, sent_len)
        sent_output, _ = self.enc_lstm(sent_packed)
        sent_output = nn.utils.rnn.pad_packed_sequence(sent_output)[0]

        # Un-sort by length
        idx_unsort = np.argsort(idx_sort)
        sent_output = sent_output.index_select(1, torch.cuda.LongTensor(idx_unsort))

        sent_output = self.proj_enc(sent_output.view(-1, 2 * self.enc_lstm_dim)).view(-1, bsize, 2 * self.enc_lstm_dim)

        # Max Pooling
        emb = torch.max(sent_output, 0)[0].squeeze(0)

        return emb


