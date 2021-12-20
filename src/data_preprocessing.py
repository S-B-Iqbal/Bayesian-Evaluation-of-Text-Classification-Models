# !pip install datasets

from datasets import load_dataset, list_datasets

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

from tqdm import tqdm

import numpy as np
import pandas as pd

reuters = load_dataset("reuters21578","ModApte") # ModApte b'coz of "A re-examination of text categorization methods" paper

train = reuters['train'] # Same as paper
test = reuters['test'] # Same as paper

train.set_format(type = "pandas")
test.set_format(type = "pandas")

df_train = train[:]
df_test = test[:]

df_train = df_train[~df_train.topics.str.len().eq(0)] # Drop Empty Topics
df_test = df_test[~df_test.topics.str.len().eq(0)] # Drop Empty Topics

cols =df_train.columns

df_train = df_train.drop([col for col in cols if col not in ['text', 'topics']], axis=1)
df_test = df_test.drop([col for col in cols if col not in ['text', 'topics']], axis=1)

X_train = df_train['text']
X_test = df_test['text']

mlb = MultiLabelBinarizer() @ Multi-Lable Problem

y_train = mlb.fit_transform(df_train.topics)
y_test = mlb.transform(df_test.topics)
