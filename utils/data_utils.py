import os
import torch
import nltk
from torch.utils.data import Dataset, DataLoader
from collections import Counter



class SNLIDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return (self.data[index]['premise'], self.data[index]['hypothesis'], self.data[index]['label'])


def load_snli_dataset(file_path, min_freq=1):
    # read the dataset file
    with open(file_path) as f:
        data = []
        for line in f:
            fields = line.strip().split('\t')
            if fields[0] == '-':  # skip examples without a gold label
                continue
            premise = nltk.word_tokenize(fields[5].lower())
            hypothesis = nltk.word_tokenize(fields[6].lower())
            label = fields[0]
            data.append({'premise': premise, 'hypothesis': hypothesis, 'label': label})

    # build the vocabulary and label dictionary
    word_counts = Counter()
    label_counts = Counter()
    for example in data:
        premise = example['premise']
        hypothesis = example['hypothesis']
        word_counts.update(premise)
        word_counts.update(hypothesis)
        label_counts.update([example['label']])

    word2idx = {'<pad>': 0, '<unk>': 1}
    for word, count in word_counts.items():
        if count >= min_freq:
            word2idx[word] = len(word2idx)

    label2idx = {}
    for label, count in label_counts.items():
        label2idx[label] = len(label2idx)

    dataset = SNLIDataset(data)

    return dataset, word2idx, label2idx

def preprocess_sentence(sentence):
    """
    Preprocesses a sentence by lowercasing and tokenizing it.

    Args:
    - sentence (str): The sentence to preprocess.

    Returns:
    - tokens (List[str]): A list of lowercase tokens.
    """
    # Lowercase the sentence
    sentence = sentence.lower()

    # Tokenize the sentence
    tokens = nltk.word_tokenize(sentence)

    return tokens


def build_vocab(data, min_freq=1):
    """
    Builds a vocabulary from the given data.

    Args:
    - data (List[Dict[str, Any]]): The data to use for building the vocabulary.
    - min_freq (int): The minimum frequency for a word to be included in the vocabulary.

    Returns:
    - vocab (Dict[str, int]): A dictionary mapping words to their indices in the vocabulary.
    """
    # Create a Counter to count the frequency of each word
    counter = Counter()

    # Iterate over the data and count the frequency of each word
    for example in data:
        premise = example['premise']
        hypothesis = example['hypothesis']
        counter.update(premise)
        counter.update(hypothesis)

    # Create the vocabulary by including only words that appear at least min_freq times
    vocab = {}
    for word, count in counter.items():
        if count >= min_freq:
            index = len(vocab)
            vocab[word] = index

    # Add special tokens to the vocabulary
    vocab['<pad>'] = len(vocab)
    vocab['<unk>'] = len(vocab)

    return vocab

def sentence_to_indices(sentence, word2idx):
    """
    Converts a sentence to a list of indices corresponding to the words in the sentence.

    Args:
    - sentence (List[str]): A list of tokens representing the sentence.
    - word2idx (Dict[str, int]): A dictionary mapping words to their indices in the vocabulary.

    Returns:
    - indices (List[int]): A list of indices corresponding to the words in the sentence.
    """
    # Convert each token in the sentence to its index in the vocabulary
    indices = [word2idx.get(token, word2idx['<unk>']) for token in sentence]

    return indices


def collate_fn(batch, word2idx, label2idx):
    premises = []
    hypotheses = []
    labels = []

    for example in batch:
        premise, hypothesis, label = example

        # Convert the premises and hypotheses to lists of indices
        premise_indices = sentence_to_indices(premise, word2idx)
        hypothesis_indices = sentence_to_indices(hypothesis, word2idx)

        # Add the processed data to the lists
        premises.append(premise_indices)
        hypotheses.append(hypothesis_indices)
        labels.append(label2idx[label])

    # Pad the premises and hypotheses
    premises = torch.nn.utils.rnn.pad_sequence(premises, batch_first=True, padding_value=word2idx['<pad>'])
    hypotheses = torch.nn.utils.rnn.pad_sequence(hypotheses, batch_first=True, padding_value=word2idx['<pad>'])

    # Convert the data to PyTorch tensors
    premises = torch.LongTensor(premises)
    hypotheses = torch.LongTensor(hypotheses)
    labels = torch.LongTensor(labels)

    return premises, hypotheses, labels

def get_train_loader(data_dir, batch_size, shuffle, word2idx, label2idx):
    # Load the development dataset
    train_dataset, _, _ = load_snli_dataset(os.path.join(data_dir, 'train.txt'))
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        collate_fn=lambda batch: collate_fn(batch, word2idx, label2idx))
    return train_loader


def get_dev_loader(data_dir, word2idx, label2idx, batch_size):
    # Load the development dataset
    dev_dataset, _, _ = load_snli_dataset(os.path.join(data_dir, 'dev.txt'))

    # Create the DataLoader
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, word2idx, label2idx))
    return dev_loader


def get_test_loader(data_dir, word2idx, label2idx, batch_size):
    # Load the test dataset
    test_dataset, _, _ = load_snli_dataset(os.path.join(data_dir, 'test.txt'))

    # Create the DataLoader
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, word2idx, label2idx))
    return test_loader
