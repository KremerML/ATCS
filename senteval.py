import sys
import numpy as np
import logging
import data_preprocess
import torch
from models import NLIClassifier, BasicEncoder, LSTMEncoder, biLSTMEncoder, biLSTMMaxPoolEncoder
import os

# Set PATHs
PATH_TO_SENTEVAL = './SentEval/'
PATH_TO_DATA = './SentEval/data/'
PRETRAINED_MODEL_PATH = './savedir/'
PATH_TO_VEC = "glove/glove.840B.300d.txt"

# Import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval

def load_model(model_class, model_path):
    model = NLIClassifier()
    model.encoder = model_class()
    model.load_state_dict(torch.load(model_path))
    return model.encoder


def prepare(params, samples):
    samples_as_strings = [' '.join(sent) for sent in samples]
    params.word2id = data_preprocess.get_word_dict(samples_as_strings)
    params.word_vec = data_preprocess.build_vocab(samples_as_strings, PATH_TO_VEC)
    params.wvec_dim = 300
    
    # Load the pretrained model
    model_class = params["model_class"]
    model_path = os.path.join(PRETRAINED_MODEL_PATH, params["model_file"])
    params["model"] = load_model(model_class, model_path)
    params["model"].eval()

    return

def batcher(params, batch):
    model = params["model"]

    # Create an instance of NLIDataset
    dummy_dataset = data_preprocess.NLIDataset([], [], [])

    embeddings = []
    for sent in batch:
        sent = ' '.join(sent)
        s1_idx, s1_length = dummy_dataset._get_sentence_indices(sent)  # Use the instance to access the method
        s1_idx = s1_idx.unsqueeze(0)
        s1_length = torch.tensor([s1_length])
        with torch.no_grad():
            sent_embedding = model((s1_idx, s1_length)).numpy()
        embeddings.append(sent_embedding)

    embeddings = np.vstack(embeddings)
    return embeddings

params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': False, 'kfold': 10}
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

models = [
    {"model_class": BasicEncoder, "model_file": "basic_enc_model.pickle"},
    {"model_class": LSTMEncoder, "model_file": "lstm_model.pickle"},
    {"model_class": biLSTMEncoder, "model_file": "bilstm_model.pickle"},
    {"model_class": biLSTMMaxPoolEncoder, "model_file": "maxpool_bilstm_model.pickle"}
]

def main():
    for model_info in models:
        print(f"Evaluating {model_info['model_class'].__name__}")
        params_senteval.update(model_info)
        se = senteval.SE(params_senteval, batcher, prepare)

        transfer_tasks = ['MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'TREC', 'MRPC', 'SICKEntailment', 'STS14']
        results = se.eval(transfer_tasks)
        print(results)

if __name__ == "__main__":
    main()
