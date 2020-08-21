#!/usr/bin/python
import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import re
import nltk
import time
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
from transformers import pipeline
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
import tensorflow_hub as hub
import itertools
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

def main():
    dataset = pd.read_csv(r'Tweets.csv')
    dataset = dataset.drop(['airline_sentiment_gold','negativereason_gold','tweet_coord'],axis=1)
    mood_count=dataset['airline_sentiment'].value_counts()
    print(mood_count)
    sns.countplot(x='airline_sentiment',data=dataset,order=['negative','neutral','positive'])
    plt.show()

    start_time = time.time()
    #substituting multiple spaces with single space
    dataset['text'] = dataset['text'].map(lambda x:re.sub(r'\s+','',str(x)))
    #removing prefixed 'b'
    dataset['text'] = dataset['text'].map(lambda x:re.sub(r'^b\s+','',str(x)))
    #remove single characters from the start
    dataset['text'] = dataset['text'].map(lambda x:re.sub(r'\^[a-zA-Z]\s+','',str(x))) 
    #remove words which are starts with @ symbols
    dataset['text'] = dataset['text'].map(lambda x:re.sub('@\w*','',str(x)))
    #remove special characters except [a-zA-Z]
    dataset['text'] = dataset['text'].map(lambda x:re.sub('[^a-zA-Z]',' ',str(x)))
    #remove link starts with https
    dataset['text'] = dataset['text'].map(lambda x:re.sub('http.*','',str(x)))
    end_time = time.time()
    #print(end_time-start_time) #print the time of data processing
    #lower case all alphabets of text column/field
    dataset['text'] = dataset['text'].map(lambda x:str(x).lower())
    start_time = time.time()

    #nltk.download('stopwords') #initially used for downloading the stopwords
    corpus = [] #initial corpus is empty
    #removing stopwords for creating the corpus
    none=dataset['text'].map(lambda x:corpus.append(' '.join([word for word in str(x).strip().split() if not word in set(stopwords.words('english'))]))) 
    end_time = time.time()
    print(end_time-start_time)
    #print(corpus[:4])
    X = pd.DataFrame(data=corpus,columns=['comment_text'])
    #extracting ground truth label
    y = dataset['airline_sentiment'].map({'neutral':1,'negative':-1,'positive':1})

    #making of data file
    data = X['comment_text']
    str1 = '\n'.join(data)
    with open("out.txt", "w") as file:
        file.write(str1)

    #using transformer based pipeline
    nlp = pipeline('sentiment-analysis')
    #transformer pipeline result generation
    y_test = np.array(y)
    y_post = []
    f = open('out.txt', 'r')
    lines = f.readlines() 
    for line in lines: 
        result = nlp(line)
        str_result = str(result[0])
        str_result = str_result.split(',')
        label = str_result[0].split(':')
        label[1] = label[1].replace("'", "")
        if(label[1].strip() == "NEGATIVE"):
            y_post.append(-1)
        if(label[1].strip() == "POSITIVE"):
            y_post.append(1)

    y_pred = np.asarray(y_post, dtype=np.float32)
    print("Transformer Pipeline Report")
    cm = confusion_matrix(y_test,y_pred)
    acc_score = accuracy_score(y_test,y_pred)
    print(classification_report(y_test,y_pred),'\n',cm,'\n',acc_score)

if __name__ == "__main__":
    main()