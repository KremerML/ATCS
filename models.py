import torch
import torch.nn as nn
from nltk.tokenize import word_tokenize
import numpy as np
import time 


class NLIClassifier(nn.Module):
    def __init__(self, config):
        super(NLIClassifier, self).__init__()
        # This architecture uses a shared sentence encoder that outputs a representation for the premise u and the hypothesis v. 
        # Once the sentence vectors are generated, 3 matching methods are applied to extract relations between u and v : 
        # (i) concatenation of the two representations (u, v); 
        # (ii) element-wise product u ∗ v; 
        # (iii) absolute element-wise difference |u − v|. 
        # The resulting vector, which captures information from both the premise and the hypothesis, 
        # is fed into a 3-class classifier consisting of multiple fully connected layers culminating in a softmax layer.

    def forward():
        raise NotImplementedError
    
    def basic_encoder():
        # averaging word embeddings to obtain sentence representations
        raise NotImplementedError
    
    def LSTM_encoder():
        # Unidirectional LSTM applied on the word embeddings
        # the last hidden state is considered as sentence representation (Section 3.2.1)
        raise NotImplementedError
    
    def biLSTM_encoder():
        # Simple bidirectional LSTM (BiLSTM)
        # last hidden state of forward and backward layers are concatenated as the sentence representation
        raise NotImplementedError
    
    def biLSTM_maxpool_encoder():
        # BiLSTM with max pooling applied to the concatenation of word-level hidden states from 
        # both directions to retrieve sentence representations (see Section 3.2.2)
        raise NotImplementedError
    
