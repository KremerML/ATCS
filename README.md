# ATCS
Advanced Topics in Computational Semantics
Natural Language Inference (NLI) with PyTorch. This repository contains an implementation of Natural Language Inference (NLI) models using PyTorch. The models are trained and evaluated on the SNLI dataset.

## Package/Installation Requirements
Python 3.6 or higher
[PyTorch](https://pytorch.org/) (latest version)
NumPy
SciPy

## Training and Evaluating a Model
Clone the repository to your local machine:
```
git clone https://github.com/KremerML/ATCS
```

Download the SNLI dataset and GloVe embeddings, and put them in the appropriate folders.

Run train.py to train a model. You can specify the model architecture, the learning rate, and other hyperparameters as command-line arguments. For example:

`python -u train.py --encoder_type LSTMEncoder --lr 0.01 --batch_size 32 --outputmodelname lstm_model.pickle --n_epochs 10`

After training, the trained model will be saved in the directory you set with `"--outputdir"`, which is set to `savedir/` by default.

The model will be automatically evaluated on the validation dataset after each epoch. The validation accuracy will be printed to the console, and the best model will be saved to the directory you set with `"--outputdir"`.

Code Structure
data_preprocess.py: Contains functions to preprocess the data, build the vocabulary, and create DataLoader objects for training, validation, and testing.
models.py: Contains the PyTorch model classes for various NLI architectures (Basic Encoder, LSTM, BiLSTM, BiLSTM with max-pooling).
train.py: Contains functions to train and evaluate the NLI models, as well as to parse command-line arguments for specifying model architectures and hyperparameters.

## Project Structure

ATCS
├─ .gitignore
├─ .vscode
│  └─ launch.json
├─ data_preprocess.py
├─ lisa_jobs
│  ├─ atcs_gpu.yml
│  ├─ install_environment.job
│  ├─ train_basicencoder.job
│  ├─ train_bilstmencoder.job
│  ├─ train_bilstmencoder_maxpool.job
│  └─ train_lstmencoder.job
├─ lisa_out
│  ├─ basicencoder_11629901.out
│  ├─ bilstm_11628674.out
│  ├─ lstm_11628673.out
│  ├─ maxpool_11628675.out
│  └─ slurm_output_11628670.out
├─ models.py
├─ README.md
├─ savedir
│  ├─ basic_enc_model.pickle
│  ├─ lstm_model.pickle
│  ├─ bilstm_model.pickle
│  └─ maxpool_bilstm_model.pickle
└─ train.py