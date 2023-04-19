import os
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from tqdm import tqdm
from nltk.tokenize import word_tokenize
import numpy as np
from datasets import load_dataset


def load_glove_embeddings(file_path):
    embeddings = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.strip().split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings[word] = vector
    return embeddings


def preprocess_data(data, save_folder):
    # Check if the save_folder exists, create it otherwise
    os.makedirs(save_folder, exist_ok=True)

    # Check if the preprocessed data has already been saved, load it and return it
    file_path = os.path.join(save_folder, "preprocessed_data.pkl")
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            return pickle.load(f)

    # Preprocess the data

    premise_tokens = [word_tokenize(sentence) for sentence in tqdm(data['premise'])]
    hypothesis_tokens = [word_tokenize(sentence) for sentence in tqdm(data['hypothesis'])]

    lc_premise_tokens = [[token.lower() for token in sentence] for sentence in tqdm(premise_tokens)]
    lc_hypothesis_tokens = [[token.lower() for token in sentence] for sentence in tqdm(hypothesis_tokens)]
    
    preprocess_data = [lc_premise_tokens, lc_hypothesis_tokens, data['label']]
    
    # Save the preprocessed data
    with open(file_path, "wb") as f:
        pickle.dump(preprocess_data, f)

    return preprocess_data

def build_vocab(train_preprocess, test_preprocess,val_preprocess, min_freq=1): 
    # Combine all data into a single list
    counter = Counter()
    vocab = {}

    # Loop over all data and update the word frequencies
    train_premise, train_hyp, _ = train_preprocess
    test_premise, test_hyp, _ = test_preprocess
    val_premise, val_hyp, _ = val_preprocess

    list_of_datasets = [train_premise, train_hyp,test_premise, test_hyp, val_premise, val_hyp]

    for dataset in tqdm(list_of_datasets):
        for sentence in dataset:
            for word in sentence:
                counter.update(word)

    for word, _ in counter.items():
        index = len(vocab)
        vocab[word] = index

    # Add special tokens to the vocabulary
    vocab['<pad>'] = len(vocab)
    vocab['<unk>'] = len(vocab)
    
    return vocab

def sentence_to_embeddings(sentence, embeddings, dim=300):
    tokens = sentence.split()
    token_embeddings = []
    for token in tokens:
        token_embeddings.append(embeddings.get(token, np.zeros(dim)))
    return token_embeddings


def collate_fn(batch, embeddings, label):
    premises = []
    hypotheses = []
    labels = []

    for example in batch:
        premise, hypothesis, label = example

        # Convert the premises and hypotheses to lists of embeddings
        premise_embeddings = sentence_to_embeddings(premise, embeddings)
        hypothesis_embeddings = sentence_to_embeddings(hypothesis, embeddings)

        # Convert the embeddings to tensors and add them to the lists
        premises.append(torch.tensor(premise_embeddings))
        hypotheses.append(torch.tensor(hypothesis_embeddings))
        labels.append(label)

    # Pad the premises and hypotheses
    premises = torch.nn.utils.rnn.pad_sequence(premises, batch_first=True, padding_value=0)
    hypotheses = torch.nn.utils.rnn.pad_sequence(hypotheses, batch_first=True, padding_value=0)

    # Convert the data to PyTorch tensors
    premises = torch.FloatTensor(premises)
    hypotheses = torch.FloatTensor(hypotheses)
    labels = torch.LongTensor(labels)

    return (premises, hypotheses, labels)



# def get_train_loader(train_preprocess, vocab, batch_size=32, shuffle=True):
#     train_premise, train_hyp, train_labels = train_preprocess

def get_train_loader(train_preprocess, vocab, batch_size=32, shuffle=True):
    train_premise, train_hyp, train_labels = train_preprocess
    train_loader = DataLoader(
        list(zip(train_premise, train_hyp, train_labels)), 
        batch_size=batch_size, 
        shuffle=shuffle, 
        collate_fn=lambda batch: collate_fn(batch, vocab, train_labels))
    return train_loader


if __name__ == '__main__':

    glove_path = "G:\GitHub\ATCS\glove\glove.840B.300d.txt"
    glove_embeddings = load_glove_embeddings(glove_path)


    train_data_raw = load_dataset('snli', split='train')
    val_data_raw = load_dataset('snli', split='validation')
    test_data_raw = load_dataset('snli', split='test')

    train_preprocess = preprocess_data(train_data_raw, save_folder="preprocessed_data/train/")
    val_preprocess = preprocess_data(val_data_raw, save_folder="preprocessed_data/val/")
    test_preprocess = preprocess_data(test_data_raw, save_folder="preprocessed_data/test/")

    vocab = build_vocab(train_preprocess, test_preprocess, val_preprocess)

    train_premise, train_hypothesis = train_preprocess[:2]
    print("train premise example {l}".format(l=train_premise[:3]))

    # trainloader = get_train_loader(train_preprocess[:2], vocab, train_preprocess[-1])
    trainloader = get_train_loader(train_preprocess, vocab)


    for i, batch in enumerate(trainloader):
        premise_batch, hypothesis_batch, label_batch = batch

        # print batch information
        print(f"Batch {i}:")
        print(f"\tPremise: {premise_batch}")
        print(f"\tHypothesis: {hypothesis_batch}")
        print(f"\tLabel: {label_batch}")




# def get_dev_loader(val_preprocess, word2idx, label2idx, batch_size):

#     # Create the DataLoader
#     dev_loader = DataLoader(
#         val_preprocess,
#         batch_size=batch_size,
#         shuffle=False,
#         collate_fn=lambda batch: collate_fn(batch, word2idx, label2idx))
#     return dev_loader


# def get_test_loader(test_preprocess, word2idx, label2idx, batch_size):
    
#     # Create the DataLoader
#     test_loader = DataLoader(
#         test_preprocess,
#         batch_size=batch_size,
#         shuffle=False,
#         collate_fn=lambda batch: collate_fn(batch, word2idx, label2idx))
#     return test_loader

# def collate_fn(batch, vocab, label):


#     premises = []
#     hypotheses = []
#     labels = []

#     for example in batch:
#         premise, hypothesis, label = example

#         # Convert the premises and hypotheses to lists of indices
#         premise_indices = sentence_to_indices(premise, vocab)
#         hypothesis_indices = sentence_to_indices(hypothesis, vocab)

#         # Add the processed data to the lists
#         premises.append(premise_indices)
#         hypotheses.append(hypothesis_indices)
#         labels.append(label)

#     # Pad the premises and hypotheses
#     premises = torch.nn.utils.rnn.pad_sequence(premises, batch_first=True, padding_value=vocab['<pad>'])
#     hypotheses = torch.nn.utils.rnn.pad_sequence(hypotheses, batch_first=True, padding_value=vocab['<pad>'])

#     # Convert the data to PyTorch tensors
#     premises = torch.LongTensor(premises)
#     hypotheses = torch.LongTensor(hypotheses)
#     labels = torch.LongTensor(labels)

#     return premises, hypotheses, labels


# def get_train_loader(train_preprocess, vocab, label, batch_size=32, shuffle=True):
    
#     train_loader = DataLoader(
#         train_preprocess, 
#         batch_size=batch_size, 
#         shuffle=shuffle, 
#         collate_fn=lambda batch: collate_fn(batch, vocab, label))
#     return train_loader
