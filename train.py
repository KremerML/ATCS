# Train the models on the SNLI dataset and use PyTorch Lightning to log the results
import os
import sys
import time
import argparse

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim

from data_preprocess import NLIDataset, get_nli, build_vocab, collate_fn
from models import NLIClassifier


parser = argparse.ArgumentParser(description='NLI training')
# paths
parser.add_argument("--nlipath", type=str, default='data/', help="SNLI data path")
parser.add_argument("--outputdir", type=str, default='savedir/', help="Output directory")
parser.add_argument("--outputmodelname", type=str, default='model.pickle')
parser.add_argument("--word_emb_path", type=str, default="glove/glove.840B.300d.txt", help="GLoVe word embedding file path")

# training
parser.add_argument("--n_epochs", type=int, default=20)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--dpout_model", type=float, default=0., help="encoder dropout")
parser.add_argument("--lr", type=float, default=0.1, help="learning rate")
parser.add_argument("--lrshrink", type=float, default=5, help="shrink factor for sgd")
parser.add_argument("--minlr", type=float, default=1e-5, help="minimum lr")

# model
parser.add_argument("--encoder_type", type=str, default='BasicEncoder', help="see list of encoders")
parser.add_argument("--enc_lstm_dim", type=int, default=2048, help="encoder nhid dimension")
parser.add_argument("--fc_dim", type=int, default=512, help="nhid of fc layers")
parser.add_argument("--n_classes", type=int, default=3, help="entailment/neutral/contradiction")

# gpu
# parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID")
parser.add_argument("--seed", type=int, default=42, help="seed")

# data
parser.add_argument("--word_emb_dim", type=int, default=300, help="word embedding dimension")

params, _ = parser.parse_known_args()

# set gpu device
torch.cuda.set_device(0)

# print parameters passed, and all parameters
print('\ntogrep : {0}\n'.format(sys.argv[1:]))
print(params)

"""
SEED
"""
np.random.seed(params.seed)
torch.manual_seed(params.seed)
torch.cuda.manual_seed(params.seed)

"""
DATA
"""
train, valid, test = get_nli(params.nlipath)
word_vec = build_vocab(train['s1'] + train['s2'] +
                       valid['s1'] + valid['s2'] +
                       test['s1'] + test['s2'], params.word_emb_path)

print("Number of words in word_vec:", len(word_vec))
print("Sample word_vec items:", list(word_vec.items())[:5])


train_dataset = NLIDataset(train, word_vec)
valid_dataset = NLIDataset(valid, word_vec)
test_dataset = NLIDataset(test, word_vec)

train_loader = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True, collate_fn=collate_fn)
valid_loader = DataLoader(valid_dataset, batch_size=params.batch_size, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=params.batch_size, shuffle=False, collate_fn=collate_fn)
"""
MODEL
"""
# model config
config_nli_model = {
    'n_words'        :  len(word_vec)         ,
    'word_emb_dim'   :  params.word_emb_dim   ,
    'enc_lstm_dim'   :  params.enc_lstm_dim   ,
    'dpout_model'    :  params.dpout_model    ,
    'fc_dim'         :  params.fc_dim         ,
    'bsize'          :  params.batch_size     ,
    'n_classes'      :  params.n_classes      ,
    'encoder_type'   :  params.encoder_type   ,
    'use_cuda'       :  True                  ,
    'vector_embeddings' : word_vec            ,
    }

# model
encoder_types = ['BasicEncoder', 'LSTMEncoder', 'biLSTMEncoder',
                 'BiLSTMMaxPoolEncoder']
assert params.encoder_type in encoder_types, "encoder_type must be in " + \
                                             str(encoder_types)
nli_net = NLIClassifier(config_nli_model)
print(nli_net)

# loss
weight = torch.FloatTensor(params.n_classes).fill_(1)
loss_fn = nn.CrossEntropyLoss(weight=weight)
loss_fn.size_average = False

# optimizer
optim_fn = optim.SGD
lr = params.lr
optimizer = optim_fn(nli_net.parameters(), lr=lr)

# cuda by default
nli_net.cuda()
loss_fn.cuda()

"""
TRAIN
"""
val_acc_best = -1e10
adam_stop = False
stop_training = False


def trainepoch(epoch):
    print('\nTRAINING : Epoch ' + str(epoch))
    nli_net.train()
    all_costs = []
    logs = []
    words_count = 0

    last_time = time.time()
    correct = 0.
    print('Learning rate : {0}'.format(lr))
    s1 = train['s1']
    s2 = train['s2']
    target = train['label']

    for stidx, (s1_batch, s1_len, s2_batch, s2_len, tgt_batch) in enumerate(train_loader):
        s1_batch, s2_batch = s1_batch.cuda(), s2_batch.cuda()
        tgt_batch = torch.LongTensor(tgt_batch).cuda()
        k = s1_batch.size(1)  # actual batch size

        # model forward
        output = nli_net((s1_batch, s1_len), (s2_batch, s2_len))

        pred = output.data.max(1)[1]
        correct += pred.long().eq(tgt_batch.data.long()).cpu().sum()
        assert len(pred) == len(s1[stidx:stidx + params.batch_size])

        # loss
        loss = loss_fn(output, tgt_batch)
        all_costs.append(loss.data[0])
        words_count += (s1_batch.nelement() + s2_batch.nelement()) / params.word_emb_dim

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient clipping (off by default)
        shrink_factor = 1
        total_norm = 0

        for p in nli_net.parameters():
            if p.requires_grad:
                p.grad.data.div_(k)  # divide by the actual batch size
                total_norm += p.grad.data.norm() ** 2
        total_norm = np.sqrt(total_norm)

        # optimizer step
        optimizer.step()

        if len(all_costs) == 100:
            logs.append('{0} ; loss {1} ; sentence/s {2} ; words/s {3} ; accuracy train : {4}'.format(
                            stidx, round(np.mean(all_costs), 2),
                            int(len(all_costs) * params.batch_size / (time.time() - last_time)),
                            int(words_count * 1.0 / (time.time() - last_time)),
                            round(100.*correct/(stidx+k), 2)))
            print(logs[-1])
            last_time = time.time()
            words_count = 0
            all_costs = []
    train_acc = round(100 * correct/len(s1), 2)
    print('results : epoch {0} ; mean accuracy train : {1}'
          .format(epoch, train_acc))
    return train_acc


def evaluate(epoch, dataloader, eval_type='valid', final_eval=False):
    nli_net.eval()
    correct = 0.
    global val_acc_best, lr, stop_training, adam_stop

    if eval_type == 'valid':
        print('\nVALIDATION : Epoch {0}'.format(epoch))

    for s1_batch, s1_len, s2_batch, s2_len, tgt_batch in dataloader:
        s1_batch, s2_batch = s1_batch.cuda(), s2_batch.cuda()
        tgt_batch = torch.LongTensor(tgt_batch).cuda()

        # model forward
        output = nli_net((s1_batch, s1_len), (s2_batch, s2_len))

        pred = output.data.max(1)[1]
        correct += pred.long().eq(tgt_batch.data.long()).cpu().sum()

    accuracy = round(100 * correct / len(dataloader.dataset), 2)
    print('results : epoch {0} ; mean accuracy {1} : {2}'.format(epoch, eval_type, accuracy))

    if eval_type == 'valid' and epoch <= params.n_epochs:
        if accuracy > val_acc_best:
            print('saving model at epoch {0}'.format(epoch))
            torch.save(nli_net.state_dict(), os.path.join(params.outputdir, params.outputmodelname))
            val_acc_best = accuracy
        else:
            lr = lr / params.lrshrink
            print('Shrinking lr by : {0}. New lr = {1}'.format(params.lrshrink,
                    lr))
            if lr < params.minlr:
                stop_training = True
    return accuracy

# Train the model
epoch = 1
while not stop_training and epoch <= params.n_epochs:
    train_acc = trainepoch(epoch)
    eval_acc = evaluate(epoch, valid_loader)
    epoch += 1

# Run the model on the test set
nli_net.load_state_dict(torch.load(os.path.join(params.outputdir, params.outputmodelname)))
print("Testing the model...")
test_acc = evaluate(1, test_loader, 'test', True)
print("Test accuracy: {}".format(test_acc))
