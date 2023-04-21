import os
import numpy as np
import torch
import json
from torch.utils.data import Dataset, DataLoader

class NLIDataset(Dataset):
    def __init__(self, data, word_vec, emb_dim=300):
        self.data = data
        self.word_vec = word_vec
        self.emb_dim = emb_dim
        label_to_idx = {'entailment': 0, 'neutral': 1, 'contradiction': 2}
        
        self.data['label'] = [label_to_idx[label] for label in self.data['label']]

    def __len__(self):
        return len(self.data['s1'])

    def __getitem__(self, idx):
        s1 = self.data['s1'][idx]
        s2 = self.data['s2'][idx]
        label = self.data['label'][idx]
        s1_idx, s1_length = self._get_sentence_indices(s1)
        s2_idx, s2_length = self._get_sentence_indices(s2)
        return s1_idx, s1_length, s2_idx, s2_length, label


    def _get_sentence_indices(self, sentence):
        tokens = sentence.split()
        length = len(tokens)
        indices = torch.zeros(length, self.emb_dim)

        for i, word in enumerate(tokens):
            if word in self.word_vec:
                indices[i] = torch.FloatTensor(self.word_vec[word])
        return indices, length


def get_word_dict(sentences):
    # create vocab of words
    word_dict = {}
    for sent in sentences:
        for word in sent.split():
            if word not in word_dict:
                word_dict[word] = ''
    word_dict['<s>'] = ''
    word_dict['</s>'] = ''
    word_dict['<p>'] = ''
    return word_dict


def get_glove(word_dict, glove_path):
    # create word_vec with glove vectors
    word_vec = {}
    with open(glove_path, encoding='utf-8') as f:
        for line in f:
            word, vec = line.split(' ', 1)
            if word in word_dict:
                word_vec[word] = list(map(float, vec.split()))
    print('Found {0}/{1} words with glove vectors'.format(
                len(word_vec), len(word_dict)))
    return word_vec



def build_vocab(sentences, glove_path):
    word_dict = get_word_dict(sentences)
    word_vec = get_glove(word_dict, glove_path)
    print('Vocab size : {0}'.format(len(word_vec)))
    return word_vec


def get_nli(data_path):
    snli_train_path = os.path.join(data_path, 'snli_1.0_train.jsonl')
    snli_valid_path = os.path.join(data_path, 'snli_1.0_dev.jsonl')
    snli_test_path = os.path.join(data_path, 'snli_1.0_test.jsonl')

    train_data = {'s1': [], 's2': [], 'label': []}
    valid_data = {'s1': [], 's2': [], 'label': []}
    test_data = {'s1': [], 's2': [], 'label': []}

    for file_path, data in zip([snli_train_path, snli_valid_path, snli_test_path], [train_data, valid_data, test_data]):
        with open(file_path, 'r') as f:
            for line in f:
                instance = json.loads(line)
                if instance['gold_label'] != '-':
                    data['s1'].append(instance['sentence1'])
                    data['s2'].append(instance['sentence2'])
                    data['label'].append(instance['gold_label'])

    return train_data, valid_data, test_data


def collate_fn(batch):
    s1, s1_lengths, s2, s2_lengths, labels = zip(*batch)
    
    s1 = torch.nn.utils.rnn.pad_sequence(s1, batch_first=True, padding_value=0)
    s2 = torch.nn.utils.rnn.pad_sequence(s2, batch_first=True, padding_value=0)
    s1_lengths = torch.tensor(s1_lengths)
    s2_lengths = torch.tensor(s2_lengths)
    labels = torch.tensor(labels, dtype=torch.long)

    return s1, s1_lengths, s2, s2_lengths, labels