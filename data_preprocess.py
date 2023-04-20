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
        s1_embed, s1_length = self._get_sentence_embedding(s1)
        s2_embed, s2_length = self._get_sentence_embedding(s2)
        return s1_embed, s1_length, s2_embed, s2_length, label

    def _get_sentence_embedding(self, sentence):
        tokens = sentence.split()
        length = len(tokens)
        embed = np.zeros((length, self.emb_dim))

        for i, word in enumerate(tokens):
            if word in self.word_vec:
                embed[i, :] = self.word_vec[word]

        return torch.from_numpy(embed).float(), length


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
                word_vec[word] = np.array(list(map(float, vec.split())))
    print('Found {0}(/{1}) words with glove vectors'.format(
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


# def get_nli(data_path):
#     s1 = {}
#     s2 = {}
#     target = {}

#     dico_label = {'entailment': 0,  'neutral': 1, 'contradiction': 2}

#     for data_type in ['train', 'dev', 'test']:
#         s1[data_type], s2[data_type], target[data_type] = {}, {}, {}
#         s1[data_type]['path'] = os.path.join(data_path, 'snli_1.0_' + data_type + '.txt')
#         s2[data_type]['path'] = os.path.join(data_path, 'snli_1.0_' + data_type + '.txt')
#         target[data_type]['path'] = os.path.join(data_path,
#                                                  'labels.' + data_type)

#         s1[data_type]['sent'] = [line.rstrip() for line in
#                                  open(s1[data_type]['path'], 'r')]
#         s2[data_type]['sent'] = [line.rstrip() for line in
#                                  open(s2[data_type]['path'], 'r')]
#         target[data_type]['data'] = np.array([dico_label[line.rstrip('\n')]
#                 for line in open(target[data_type]['path'], 'r')])

#         assert len(s1[data_type]['sent']) == len(s2[data_type]['sent']) == \
#             len(target[data_type]['data'])

#         print('** {0} DATA : Found {1} pairs of {2} sentences.'.format(
#                 data_type.upper(), len(s1[data_type]['sent']), data_type))

#     train = {'s1': s1['train']['sent'], 's2': s2['train']['sent'],
#              'label': target['train']['data']}
#     dev = {'s1': s1['dev']['sent'], 's2': s2['dev']['sent'],
#            'label': target['dev']['data']}
#     test = {'s1': s1['test']['sent'], 's2': s2['test']['sent'],
#             'label': target['test']['data']}
#     return train, dev, test

# def get_nli_dataloader(data, word_vec, batch_size=32, shuffle=True, num_workers=0):
#     dataset = NLIDataset(data, word_vec)
#     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
#                             collate_fn=collate_fn)
#     return dataloader

def collate_fn(batch):
    s1, s1_lengths, s2, s2_lengths, labels = zip(*batch)
    s1_lengths, s1_perm_idx = torch.tensor(s1_lengths).sort(descending=True)
    s2_lengths, s2_perm_idx = torch.tensor(s2_lengths).sort(descending=True)

    s1 = torch.nn.utils.rnn.pad_sequence([s1[i] for i in s1_perm_idx], batch_first=True)
    s2 = torch.nn.utils.rnn.pad_sequence([s2[i] for i in s2_perm_idx], batch_first=True)
    labels = torch.tensor(labels, dtype=torch.long)

    return s1, s1_lengths, s2, s2_lengths, labels


# Usage:
# train_data, dev_data, test_data = get_nli(data_path)
# word_vec = build_vocab(train_data['s1'] + train_data['s2'], glove_path)
# train_loader = get_nli_dataloader(train_data, word_vec)
# dev_loader = get_nli_dataloader(dev_data, word_vec)
# test_loader = get_nli_dataloader(test_data, word_vec)
