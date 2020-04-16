import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

def importdata():
    balance_data = pd.read_csv('balance-scale.csv',
                               sep=',',header=None)
    print("Dataset Length: ",len(balance_data))
    print("Dataset Shape: ",balance_data.shape)

    return balance_data

def splitdata(balance_data):
    #Separating the target variable 
    X = balance_data.values[:, 1:5]
    Y = balance_data.values[:, 0]

    #Splitting the dataset into train and test 
    X_train, X_test, y_train, y_test = train_test_split(  
    X, Y, test_size = 0.3, random_state = 100) 
    return X, Y, X_train, X_test, y_train, y_test    
        
def main():
    data = importdata()
    X, Y, X_train, X_test, y_train, y_test = splitdata(data)

    np.random.seed(0) #makes the random numbers predictable

    clf = RandomForestClassifier(n_jobs=2,random_state=0) #n_jobs used for paralleling

    clf.fit(X_train,y_train)

    print("Accuracy of balance-scale data using random forest:")
    print(accuracy_score(clf.predict(X_test),y_test))
    

if __name__ == "__main__":
    main()
