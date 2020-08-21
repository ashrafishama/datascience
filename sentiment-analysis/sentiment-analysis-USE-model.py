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

#data structure for storing roc related results
result_table = pd.DataFrame(columns=['classifiers', 'fpr','tpr','auc'])
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

#train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

#creating data structure to be used for 'USE' model
X_train = X_train.values.tolist()
X_test = X_test.values.tolist()
X_train = list(itertools.chain.from_iterable(X_train))
X_test = list(itertools.chain.from_iterable(X_test))

#using Universal Sentence Encoder
module_url = "https://tfhub.dev/google/universal-sentence-encoder/2" 
#@param ["https://tfhub.dev/google/universal-sentence-encoder/2"
# "https://tfhub.dev/google/universal-sentence-encoder-large/3"]
# Import the Universal Sentence Encoder's TF Hub module
embed = hub.Module(module_url)
# Reduce logging output.
tf.logging.set_verbosity(tf.logging.ERROR)
with tf.Session() as session:
  session.run([tf.global_variables_initializer(), tf.tables_initializer()])
  training_sentence_embeddings = session.run(embed(X_train))
  testing_sentence_embeddings =  session.run(embed(X_test))

'''
#standard scaling
scaler = StandardScaler()
X_train_word_feature = scaler.fit_transform(X_train_word_feature)
X_test_word_feature = scaler.fit_transform(X_test_word_feature)
'''

#applying PCA
pca_model = PCA(n_components=20)
#for USE model
pca_model.fit(training_sentence_embeddings)
X_train_word_feature_knn = pca_model.transform(training_sentence_embeddings)
X_test_word_feature_knn = pca_model.transform(testing_sentence_embeddings)

#applying kNN
k = 5
knn_model = KNeighborsClassifier(n_neighbors=k)
knn_model.fit(X_train_word_feature_knn, y_train)
y_pred = knn_model.predict(X_test_word_feature_knn)
yproba = knn_model.predict_proba(X_test_word_feature_knn)[::,1]
fpr, tpr, _ = roc_curve(y_test,yproba)
auc = roc_auc_score(y_test, yproba)
result_table = result_table.append({'classifiers':'kNN',
'fpr':fpr,'tpr':tpr, 'auc':auc}, ignore_index=True)

acc_score = accuracy_score(y_test,y_pred)
print("kNN Classification Report")
cm = confusion_matrix(y_test,y_pred)
print("Accuracy of kNN",acc_score)
print(classification_report(y_test,y_pred),'\n',cm,'\n',acc_score)

#fitting SVM to the model
clf_SVM = SVC(gamma='auto',probability=True)
#for USE model
clf_SVM.fit(training_sentence_embeddings,y_train)
y_pred = clf_SVM.predict(testing_sentence_embeddings)
yproba = clf_SVM.predict_proba(testing_sentence_embeddings)[::,1]
fpr, tpr, _ = roc_curve(y_test,yproba)
auc = roc_auc_score(y_test, yproba)
result_table = result_table.append({'classifiers':'SVM',
'fpr':fpr,'tpr':tpr, 'auc':auc}, ignore_index=True)

acc_score = accuracy_score(y_test,y_pred)
print("SVM Classification Report")
cm = confusion_matrix(y_test,y_pred)
print("Accuracy of SVM:",acc_score)
print(classification_report(y_test,y_pred),'\n',cm,'\n',acc_score)


#fitting Random forest to the model
rfc = RandomForestClassifier(n_estimators=100, random_state=0)
#for USE model
rfc.fit(training_sentence_embeddings,y_train)
rfc_predict = rfc.predict(testing_sentence_embeddings)
yproba = rfc.predict_proba(testing_sentence_embeddings)[::,1]
fpr, tpr, _ = roc_curve(y_test,yproba)
auc = roc_auc_score(y_test, yproba)
result_table = result_table.append({'classifiers':'Random Forest',
'fpr':fpr,'tpr':tpr, 'auc':auc}, ignore_index=True)

print("Random Forest Classifier Report")
cm = confusion_matrix(y_test,rfc_predict)
acc_score = accuracy_score(y_test,rfc_predict)
print(classification_report(y_test,rfc_predict),'\n',cm,'\n',acc_score)

#fitting Logistic Regression to the model
classifier = LogisticRegression()
#using USE model
classifier.fit(training_sentence_embeddings,y_train)
y_pred = classifier.predict(testing_sentence_embeddings)
yproba = classifier.predict_proba(testing_sentence_embeddings)[::,1]
fpr, tpr, _ = roc_curve(y_test,yproba)
auc = roc_auc_score(y_test, yproba)
result_table = result_table.append({'classifiers':'Logistic Regression',
'fpr':fpr,'tpr':tpr, 'auc':auc}, ignore_index=True)

print("Logistic Regression Classifier Report")
cm = confusion_matrix(y_test,y_pred)
acc_score = accuracy_score(y_test,y_pred)
print(classification_report(y_test,y_pred),'\n',cm,'\n',acc_score)

# Set name of the classifiers as index labels
result_table.set_index('classifiers', inplace=True)
#drawing ROC
fig = plt.figure(figsize=(8,6))
for i in result_table.index:
    plt.plot(result_table.loc[i]['fpr'], 
             result_table.loc[i]['tpr'], 
             label="{}, AUC={:.3f}".format(i, result_table.loc[i]['auc']))
    
plt.plot([0,1], [0,1], color='orange', linestyle='--') #the corner line
plt.xticks(np.arange(0.0, 1.1, step=0.1))
plt.xlabel("False Positive Rate", fontsize=15)
plt.yticks(np.arange(0.0, 1.1, step=0.1))
plt.ylabel("True Positive Rate", fontsize=15)
plt.title('ROC Curve Analysis', fontweight='bold', fontsize=15)
plt.legend(prop={'size':13}, loc='lower right')
plt.show()