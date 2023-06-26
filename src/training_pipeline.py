import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.pipeline import Pipeline 
import pickle

from src.preprocessing import DataCleaning
from src.preprocessing import Stemmer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

df=pd.read_csv(os.path.join('notebooks/data','text_emotion.csv'))

#train test split

X_train, X_test, y_train, y_test = train_test_split(df['text'], df['emotion'], test_size=0.2, random_state=42,stratify=df['emotion'])
tfidf=TfidfVectorizer(max_features=2000,ngram_range=(1,2))
X_train_tfidf=tfidf.fit_transform(X_train)
X_test_tfidf=tfidf.transform(X_test)

lr=LogisticRegression(solver='liblinear')
lr.fit(X_train_tfidf,y_train)

y_pred_test=lr.predict(X_test_tfidf)
y_pred_train=lr.predict(X_train_tfidf)

print("TRAIN: ",accuracy_score(y_train,y_pred_train)," ","TEST: ",accuracy_score(y_test,y_pred_test))


classifier=Pipeline(steps=[
    ('cleaner',DataCleaning()),
    ('vectorizer',tfidf),
    ('model',lr)

])



with open('model_classifier.pickle', 'wb') as picklefile:
    pickle.dump(classifier, picklefile)