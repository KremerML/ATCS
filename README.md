# ATCS
Advanced Topics in Computational Semantics
Natural Language Inference (NLI) with PyTorch. This repository contains an implementation of Natural Language Inference (NLI) models using PyTorch. The models are trained and evaluated on the SNLI dataset.

## Package/Installation Requirements
- Python 3.6 or higher
- [PyTorch](https://pytorch.org/) (latest version)
- NumPy
- SciPy

## Training and Evaluating a Model
Clone the repository to your local machine:
```
git clone https://github.com/KremerML/ATCS
```

Download the SNLI dataset and GloVe embeddings, and put them in the appropriate folders.

Run train.py to train a model. You can specify the model architecture, the learning rate, and other hyperparameters as command-line arguments. For example:

```
python -u train.py --encoder_type LSTMEncoder --lr 0.01 --batch_size 32 --outputmodelname lstm_model.pickle --n_epochs 10
```

After training, the trained model will be saved in the directory you set with `"--outputdir"`, which is set to `savedir/` by default.

The model will be automatically evaluated on the validation dataset after each epoch. The validation accuracy will be printed to the console, and the best model will be saved to the directory you set with `"--outputdir"`.

## Code Structure
**data_preprocess.py**: Contains functions to preprocess the data, build the vocabulary, and create DataLoader objects for training, validation, and testing.

**models.py**: Contains the PyTorch model classes for various NLI architectures (Basic Encoder, LSTM, BiLSTM, BiLSTM with max-pooling).

**train.py**: Contains functions to train and evaluate the NLI models, as well as to parse command-line arguments for specifying model architectures and hyperparameters.

**senteval.py**: Contains functions to evaluate the NLI models on the SNLI dataset using [SentEval](https://github.com/facebookresearch/SentEval) framework.

**results_analysis.ipynb**: Jupyter notebook for analyzing the results of the NLI models.

## Project Structure
ATCS
├─ .gitignore

├─ .vscode

│  └─ launch.json

├─ data_preprocess.py

├─ models.py

├─ README.md

├─ results_analysis.ipynb

├─ senteval.py

└─ train.py