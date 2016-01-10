__author__ = 'neuralconcept'

import pandas as pd
import os
from nltk.corpus import stopwords
import nltk.data
import logging
import dill
import numpy as np  # Make sure that numpy is imported
from gensim.models import Word2Vec
from Brain_LSTM import Model
from WordUtility import WordUtility
from theano_lstm import LSTM

f = file('Brain.save', 'rb')
v = file('BrainVoca.save', 'rb')
model = dill.load(f)
vocab = dill.load(v)

f.close()
v.close()
print(vocab(model.greedy_fun(vocab.questionGeneration("But why?"))))

