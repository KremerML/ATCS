import os
import torch
import unittest
import nltk
from collections import Counter
from torch.utils.data import DataLoader
from unittest.mock import patch, MagicMock

from utils.data_utils import SNLIDataset, load_snli_dataset, preprocess_sentence, build_vocab, sentence_to_indices, collate_fn, get_train_loader

class TestSNLIDataset(unittest.TestCase):

    def test_SNLI_dataset_length(self):
        data = [
            {'premise': ['this', 'is', 'a', 'premise'], 'hypothesis': ['this', 'is', 'a', 'hypothesis'], 'label': 'entailment'},
            {'premise': ['another', 'premise'], 'hypothesis': ['another', 'hypothesis'], 'label': 'contradiction'},
            {'premise': ['the', 'third', 'premise'], 'hypothesis': ['the', 'third', 'hypothesis'], 'label': 'neutral'}
        ]
        dataset = SNLIDataset(data)
        self.assertEqual(len(dataset), 3)
        print("test_SNLI_dataset_length passed.")

    def test_SNLI_dataset_getitem(self):
        data = [
            {'premise': ['this', 'is', 'a', 'premise'], 'hypothesis': ['this', 'is', 'a', 'hypothesis'], 'label': 'entailment'},
            {'premise': ['another', 'premise'], 'hypothesis': ['another', 'hypothesis'], 'label': 'contradiction'},
            {'premise': ['the', 'third', 'premise'], 'hypothesis': ['the', 'third', 'hypothesis'], 'label': 'neutral'}
        ]
        dataset = SNLIDataset(data)
        self.assertEqual(dataset[0], (['this', 'is', 'a', 'premise'], ['this', 'is', 'a', 'hypothesis'], 'entailment'))
        self.assertEqual(dataset[1], (['another', 'premise'], ['another', 'hypothesis'], 'contradiction'))
        self.assertEqual(dataset[2], (['the', 'third', 'premise'], ['the', 'third', 'hypothesis'], 'neutral'))
        print("test_SNLI_dataset_getitem passed.")

    def test_load_snli_dataset(self):
        with patch('builtins.open', new_callable=MagicMock) as mock_open:
            mock_file = MagicMock()
            mock_file.__enter__.return_value = mock_file
            mock_file.__iter__.return_value = [
                '-\t-\t-\t-\t-\tthis is a premise\tthis is a hypothesis\n',
                'contradiction\t-\t-\t-\t-\tanother premise\tanother hypothesis\n',
                'neutral\t-\t-\t-\t-\tthe third premise\tthe third hypothesis\n'
            ]
            mock_open.return_value = mock_file

            dataset, word2idx, label2idx = load_snli_dataset('path/to/data')
            self.assertEqual(len(dataset), 2)
            self.assertEqual(len(word2idx), 8)
            self.assertEqual(len(label2idx), 2)
        print("test_load_snli_dataset passed.")

    def test_preprocess_sentence(self):
        sentence = 'This is a test sentence.'
        tokens = preprocess_sentence(sentence)
        self.assertEqual(tokens, ['this', 'is', 'a', 'test', 'sentence', '.'])
        print("test_preprocess_sentence passed.")

    def test_build_vocab(self):
        data = [
            {'premise': ['this', 'is', 'a', 'premise'], 'hypothesis': ['this', 'is', 'a', 'hypothesis'], 'label': 'entailment'},
            {'premise': ['another', 'premise'], 'hypothesis': ['another', 'hypothesis'], 'label': 'contradiction'},
            {'premise': ['the', 'third', 'premise'], 'hypothesis': ['the', 'third', 'hypothesis'], 'label': 'neutral'}
            ]
        _, word2idx, idx2word = load_snli_dataset()
        self.assertEqual(len(word2idx), 8)
        self.assertEqual(len(idx2word), 8)
        self.assertEqual(word2idx['this'], 0)
        self.assertEqual(word2idx['a'], 2)
        self.assertEqual(idx2word[4], 'hypothesis')
        print("test_build_vocab passed.")

    def test_sentence_to_indices(self):
        sentence = ['this', 'is', 'a', 'test', 'sentence', '.']
        word2idx = {'this': 0, 'is': 1, 'a': 2, 'test': 3, 'sentence': 4, '.': 5}
        indices = sentence_to_indices(sentence, word2idx)
        self.assertEqual(indices, [0, 1, 2, 3, 4, 5])
        print("test_sentence_to_indices passed.")

    def test_collate_fn(self):
        data = [
            (['this', 'is', 'a', 'premise'], ['this', 'is', 'a', 'hypothesis'], 'entailment'),
            (['another', 'premise'], ['another', 'hypothesis'], 'contradiction'),
            (['the', 'third', 'premise'], ['the', 'third', 'hypothesis'], 'neutral')
        ]
        word2idx = {'this': 0, 'is': 1, 'a': 2, 'premise': 3, 'hypothesis': 4, 'another': 5, 'the': 6, 'third': 7, 'neutral': 8, 'entailment': 9, 'contradiction': 10}
        premise, hypothesis, label = collate_fn(data, word2idx)
        self.assertEqual(torch.Tensor(premise), torch.Tensor([[0, 1, 2, 3], [5, 3], [6, 7, 3]]))
        self.assertEqual(torch.Tensor(hypothesis), torch.Tensor([[0, 1, 2, 4], [5, 4], [6, 7, 4]]))
        self.assertEqual(torch.Tensor(label), torch.Tensor([9, 10, 8]))
        print("test_collate_fn passed.")

    def test_get_train_loader(self):
        data = [
            {'premise': ['this', 'is', 'a', 'premise'], 'hypothesis': ['this', 'is', 'a', 'hypothesis'], 'label': 'entailment'},
            {'premise': ['another', 'premise'], 'hypothesis': ['another', 'hypothesis'], 'label': 'contradiction'},
            {'premise': ['the', 'third', 'premise'], 'hypothesis': ['the', 'third', 'hypothesis'], 'label': 'neutral'}
        ]
        word2idx = {'this': 0, 'is': 1, 'a': 2, 'premise': 3, 'hypothesis': 4, 'another': 5, 'the': 6, 'third': 7, 'neutral': 8, 'entailment': 9, 'contradiction': 10}
        dataset = SNLIDataset(data, word2idx=word2idx)
        train_loader = get_train_loader(dataset, batch_size=2)

        self.assertEqual(len(train_loader), 2)
        batches = []
        for batch in train_loader:
            batches.append(batch)
            self.assertEqual(len(batch), 3)
            self.assertEqual(len(batch[0]), 2)
            self.assertEqual(len(batch[1]), 2)
            self.assertEqual(len(batch[2]), 2)

        self.assertEqual(len(batches), 2)
        print("test_get_train_loader passed.")

if __name__ == '__main__':
    unittest.main()
